"""LLM-as-judge scoring via the Groq API, with judge-variance reporting."""

import json
import os
import re
import statistics
import time
from typing import Any

from groq import Groq

from evalforge.schema import EvalResult, ModelOutput, TestCase
from evalforge.scorers.base import Scorer

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_VARIANCE_THRESHOLD = 0.15
DEFAULT_NUM_CALLS = 3
MAX_RETRIES = 2

SYSTEM_PROMPT = """You are a strict evaluator of AI model outputs.

Score the output on correctness and quality as a float between 0.0 and 1.0,
where 0.0 is completely wrong or useless and 1.0 is fully correct and high
quality. When a reference answer is given, judge primarily on agreement with
it; differences in wording or formatting alone should not lower the score.

Respond with ONLY a JSON object in exactly this shape, with no extra keys,
no markdown fences, and no prose outside the JSON:
{"score": <float between 0.0 and 1.0>, "reasoning": "<one or two sentences>"}"""


class JudgeError(RuntimeError):
    """Raised when the judge cannot produce a usable score."""


def _build_user_prompt(test_case: TestCase, output: ModelOutput) -> str:
    reference_block = (
        f"REFERENCE ANSWER:\n{test_case.reference}"
        if test_case.reference is not None
        else "REFERENCE ANSWER:\n(none provided - judge on correctness and quality alone)"
    )
    return (
        f"INPUT:\n{test_case.input}\n\n"
        f"{reference_block}\n\n"
        f"MODEL OUTPUT:\n{output.output}"
    )


def _extract_json(content: str) -> dict[str, Any]:
    """Parse a JSON object from judge output, tolerating fences or stray prose."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match is None:
            raise JudgeError(f"no JSON object found in judge response: {content!r}")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise JudgeError(
                f"malformed JSON in judge response: {content!r}"
            ) from exc

    if not isinstance(parsed, dict):
        raise JudgeError(f"judge response was not a JSON object: {content!r}")
    return parsed


def _parse_judgement(content: str) -> tuple[float, str]:
    parsed = _extract_json(content)

    if "score" not in parsed:
        raise JudgeError(f"judge response missing 'score': {content!r}")

    score = parsed["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise JudgeError(f"judge 'score' was not a number: {content!r}")
    # Out-of-range means the judge ignored the 0-1 scale (e.g. answered on a
    # 0-10 one). Rescaling would guess at intent, so treat it as malformed.
    if not 0.0 <= score <= 1.0:
        raise JudgeError(f"judge 'score' outside [0, 1]: {content!r}")

    reasoning = parsed.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise JudgeError(f"judge 'reasoning' was not a string: {content!r}")

    return float(score), reasoning


class LLMJudgeScorer(Scorer):
    """Scores output by asking a Groq-hosted LLM to grade it several times.

    The judge is called `num_calls` times independently per test case. The
    score is the plain mean of those calls; the spread across them is reported
    as `std_dev` and flagged via `high_variance` when it exceeds
    `variance_threshold`. Outliers are never dropped and the mean is never
    adjusted - a high-variance flag is a signal for a human to interpret, not
    a correction the harness applies on its own.
    """

    name = "llm_judge"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
        num_calls: int = DEFAULT_NUM_CALLS,
        temperature: float = 1.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.variance_threshold = variance_threshold
        self.num_calls = num_calls
        # Judge calls are deliberately sampled at a non-zero temperature so the
        # reported variance reflects real judge instability.
        self.temperature = temperature

        if client is not None:
            self._client = client
        else:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set; export it or pass a client explicitly"
                )
            self._client = Groq(api_key=api_key)

    def _judge_once(self, user_prompt: str) -> tuple[float, str]:
        last_error: JudgeError | None = None
        for _ in range(MAX_RETRIES + 1):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            try:
                return _parse_judgement(content)
            except JudgeError as exc:
                last_error = exc
        raise JudgeError(
            f"judge returned unusable output after {MAX_RETRIES + 1} attempts"
        ) from last_error

    def score(self, test_case: TestCase, output: ModelOutput) -> EvalResult:
        user_prompt = _build_user_prompt(test_case, output)

        start = time.perf_counter()
        scores: list[float] = []
        reasonings: list[str] = []
        for _ in range(self.num_calls):
            call_score, reasoning = self._judge_once(user_prompt)
            scores.append(call_score)
            reasonings.append(reasoning)
        latency_ms = (time.perf_counter() - start) * 1000

        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        return EvalResult(
            test_case_id=test_case.id,
            method=self.name,
            score=statistics.mean(scores),
            latency_ms=latency_ms,
            raw_output={
                "scores": scores,
                "std_dev": std_dev,
                "reasonings": reasonings,
                "high_variance": std_dev > self.variance_threshold,
            },
        )
