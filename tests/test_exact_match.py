import pytest

from evalforge.schema import ModelOutput, TestCase
from evalforge.scorers.exact_match import ExactMatchScorer


@pytest.fixture
def scorer() -> ExactMatchScorer:
    return ExactMatchScorer()


def test_identical_strings_score_one(scorer):
    case = TestCase(id="c1", input="q", reference="Paris")
    output = ModelOutput(test_case_id="c1", output="Paris")
    result = scorer.score(case, output)
    assert result.score == 1.0
    assert result.method == "exact_match"


def test_case_and_whitespace_insensitive(scorer):
    case = TestCase(id="c1", input="q", reference="  Paris  ")
    output = ModelOutput(test_case_id="c1", output="paris")
    result = scorer.score(case, output)
    assert result.score == 1.0


def test_completely_different_strings_score_zero(scorer):
    case = TestCase(id="c1", input="q", reference="Paris")
    output = ModelOutput(test_case_id="c1", output="London")
    result = scorer.score(case, output)
    assert result.score == 0.0
    assert result.raw_output["jaccard_token_overlap"] == 0.0


def test_near_miss_has_zero_score_but_high_jaccard_diagnostic(scorer):
    case = TestCase(id="c1", input="q", reference="The cat sat on the mat")
    output = ModelOutput(test_case_id="c1", output="The cat sat on a mat")
    result = scorer.score(case, output)
    assert result.score == 0.0
    # ref tokens: {the, cat, sat, on, mat} (5); output tokens: {the, cat, sat, on, a, mat} (6)
    # intersection = 5, union = 6
    assert result.raw_output["jaccard_token_overlap"] == pytest.approx(5 / 6)


def test_missing_reference_raises(scorer):
    case = TestCase(id="c1", input="q", reference=None)
    output = ModelOutput(test_case_id="c1", output="anything")
    with pytest.raises(ValueError):
        scorer.score(case, output)


def test_latency_is_recorded(scorer):
    case = TestCase(id="c1", input="q", reference="Paris")
    output = ModelOutput(test_case_id="c1", output="Paris")
    result = scorer.score(case, output)
    assert result.latency_ms >= 0.0
