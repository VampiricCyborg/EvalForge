import json
import statistics
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from evalforge.schema import ModelOutput, TestCase
from evalforge.scorers.llm_judge import JudgeError, LLMJudgeScorer


def _response(content: str) -> SimpleNamespace:
    """A stand-in for a Groq chat completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _judgement(score: float, reasoning: str = "looks fine") -> SimpleNamespace:
    return _response(json.dumps({"score": score, "reasoning": reasoning}))


def _client(*contents) -> MagicMock:
    """A mock Groq client returning the given responses in order."""
    client = MagicMock()
    client.chat.completions.create.side_effect = list(contents)
    return client


@pytest.fixture
def case() -> TestCase:
    return TestCase(id="c1", input="What is 2+2?", reference="4")


@pytest.fixture
def output() -> ModelOutput:
    return ModelOutput(test_case_id="c1", output="4")


def test_scores_are_averaged_over_three_calls(case, output):
    client = _client(_judgement(0.8), _judgement(0.9), _judgement(0.7))
    scorer = LLMJudgeScorer(client=client)

    result = scorer.score(case, output)

    assert client.chat.completions.create.call_count == 3
    assert result.method == "llm_judge"
    assert result.score == pytest.approx(0.8)
    assert result.raw_output["scores"] == [0.8, 0.9, 0.7]
    assert result.raw_output["reasonings"] == ["looks fine"] * 3
    assert result.latency_ms >= 0.0


def test_consistent_scores_are_not_flagged(case, output):
    client = _client(_judgement(0.8), _judgement(0.82), _judgement(0.78))
    scorer = LLMJudgeScorer(client=client)

    result = scorer.score(case, output)

    assert result.raw_output["std_dev"] < 0.15
    assert result.raw_output["high_variance"] is False


def test_high_variance_is_flagged_without_adjusting_score(case, output):
    scores = [0.1, 0.9, 0.5]
    client = _client(*[_judgement(s) for s in scores])
    scorer = LLMJudgeScorer(client=client)

    result = scorer.score(case, output)

    assert result.raw_output["high_variance"] is True
    assert result.raw_output["std_dev"] == pytest.approx(statistics.stdev(scores))
    # The outlier is neither dropped nor down-weighted: the score is the plain mean.
    assert result.raw_output["scores"] == scores
    assert result.score == pytest.approx(0.5)


def test_variance_threshold_is_configurable(case, output):
    client = _client(_judgement(0.8), _judgement(0.82), _judgement(0.78))
    scorer = LLMJudgeScorer(client=client, variance_threshold=0.001)

    result = scorer.score(case, output)

    assert result.raw_output["high_variance"] is True


def test_malformed_json_is_retried(case, output):
    client = _client(
        _response("not json at all"),
        _judgement(0.6),
        _judgement(0.6),
        _judgement(0.6),
    )
    scorer = LLMJudgeScorer(client=client)

    result = scorer.score(case, output)

    assert client.chat.completions.create.call_count == 4  # 1 retry + 3 good calls
    assert result.score == pytest.approx(0.6)


def test_json_wrapped_in_markdown_fences_is_parsed(case, output):
    fenced = '```json\n{"score": 0.5, "reasoning": "partial"}\n```'
    client = _client(_response(fenced), _response(fenced), _response(fenced))
    scorer = LLMJudgeScorer(client=client)

    result = scorer.score(case, output)

    assert result.score == pytest.approx(0.5)
    assert client.chat.completions.create.call_count == 3


def test_score_outside_range_is_treated_as_malformed(case, output):
    # A judge answering on a 0-10 scale must not be silently clamped to 1.0.
    client = _client(_judgement(7), _judgement(0.4), _judgement(0.4), _judgement(0.4))
    scorer = LLMJudgeScorer(client=client)

    result = scorer.score(case, output)

    assert result.score == pytest.approx(0.4)


def test_missing_score_key_is_treated_as_malformed(case, output):
    client = _client(
        _response('{"reasoning": "forgot the score"}'),
        _judgement(0.3),
        _judgement(0.3),
        _judgement(0.3),
    )
    scorer = LLMJudgeScorer(client=client)

    assert scorer.score(case, output).score == pytest.approx(0.3)


def test_persistently_malformed_output_raises(case, output):
    client = _client(*[_response("garbage")] * 3)
    scorer = LLMJudgeScorer(client=client)

    with pytest.raises(JudgeError):
        scorer.score(case, output)

    assert client.chat.completions.create.call_count == 3  # initial + 2 retries


def test_works_without_a_reference(output):
    case = TestCase(id="c1", input="Write a haiku.", reference=None)
    client = _client(_judgement(0.9), _judgement(0.9), _judgement(0.9))
    scorer = LLMJudgeScorer(client=client)

    result = scorer.score(case, output)

    assert result.score == pytest.approx(0.9)
    prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "none provided" in prompt


def test_prompt_includes_input_reference_and_output(case, output):
    client = _client(_judgement(1.0), _judgement(1.0), _judgement(1.0))
    scorer = LLMJudgeScorer(client=client)

    scorer.score(case, output)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["model"] == "llama-3.3-70b-versatile"
    user_prompt = kwargs["messages"][1]["content"]
    assert "What is 2+2?" in user_prompt
    assert "4" in user_prompt


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        LLMJudgeScorer()


def test_num_calls_is_configurable(case, output):
    client = _client(_judgement(0.5), _judgement(0.7))
    scorer = LLMJudgeScorer(client=client, num_calls=2)

    result = scorer.score(case, output)

    assert client.chat.completions.create.call_count == 2
    assert result.score == pytest.approx(0.6)


def test_single_call_reports_zero_variance(case, output):
    client = _client(_judgement(0.5))
    scorer = LLMJudgeScorer(client=client, num_calls=1)

    result = scorer.score(case, output)

    assert result.raw_output["std_dev"] == 0.0
    assert result.raw_output["high_variance"] is False
