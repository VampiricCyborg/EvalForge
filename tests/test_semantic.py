import pytest

from evalforge.schema import ModelOutput, TestCase
from evalforge.scorers.semantic import SemanticSimilarityScorer


@pytest.fixture(scope="module")
def scorer() -> SemanticSimilarityScorer:
    # Loads the local all-MiniLM-L6-v2 model once and reuses it across tests.
    return SemanticSimilarityScorer()


def test_identical_strings_score_near_one(scorer):
    case = TestCase(id="c1", input="q", reference="The cat sat on the mat.")
    output = ModelOutput(test_case_id="c1", output="The cat sat on the mat.")
    result = scorer.score(case, output)
    assert result.score > 0.98
    assert result.method == "semantic_similarity"


def test_paraphrase_scores_high(scorer):
    case = TestCase(id="c1", input="q", reference="The cat sat on the mat.")
    output = ModelOutput(test_case_id="c1", output="A cat was sitting on the mat.")
    result = scorer.score(case, output)
    assert result.score > 0.6


def test_unrelated_strings_score_low(scorer):
    case = TestCase(id="c1", input="q", reference="The cat sat on the mat.")
    output = ModelOutput(
        test_case_id="c1",
        output="Quantum entanglement enables correlated measurements.",
    )
    result = scorer.score(case, output)
    assert result.score < 0.35


def test_score_is_bounded(scorer):
    case = TestCase(id="c1", input="q", reference="Hello world")
    output = ModelOutput(test_case_id="c1", output="Goodbye moon")
    result = scorer.score(case, output)
    assert 0.0 <= result.score <= 1.0
    assert "raw_cosine_similarity" in result.raw_output


def test_missing_reference_raises(scorer):
    case = TestCase(id="c1", input="q", reference=None)
    output = ModelOutput(test_case_id="c1", output="anything")
    with pytest.raises(ValueError):
        scorer.score(case, output)
