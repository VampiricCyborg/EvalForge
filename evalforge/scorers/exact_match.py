"""Binary exact-match scoring, with a fuzzy token-overlap diagnostic."""

import re
import time

from evalforge.schema import EvalResult, ModelOutput, TestCase
from evalforge.scorers.base import Scorer


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _jaccard(a: str, b: str) -> float:
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


class ExactMatchScorer(Scorer):
    """Scores 1.0 if output matches reference after normalization, else 0.0.

    The primary score is strictly binary. A Jaccard token-overlap score is
    computed as a diagnostic and stashed in `raw_output` so near-misses are
    visible in reports without affecting the score itself.
    """

    name = "exact_match"

    def score(self, test_case: TestCase, output: ModelOutput) -> EvalResult:
        if test_case.reference is None:
            raise ValueError(
                f"test_case '{test_case.id}' has no reference; "
                f"{self.name} requires one to score against"
            )

        start = time.perf_counter()
        normalized_reference = _normalize(test_case.reference)
        normalized_output = _normalize(output.output)
        is_match = normalized_reference == normalized_output
        jaccard_overlap = _jaccard(normalized_reference, normalized_output)
        latency_ms = (time.perf_counter() - start) * 1000

        return EvalResult(
            test_case_id=test_case.id,
            method=self.name,
            score=1.0 if is_match else 0.0,
            latency_ms=latency_ms,
            raw_output={
                "normalized_reference": normalized_reference,
                "normalized_output": normalized_output,
                "jaccard_token_overlap": jaccard_overlap,
            },
        )
