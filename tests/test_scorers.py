import pytest

from evalforge.schema import EvalResult, ModelOutput, TestCase
from evalforge.scorers.base import Scorer


def test_scorer_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Scorer()


def test_concrete_scorer_subclass_works(test_case, model_output):
    class ExactMatchScorer(Scorer):
        name = "exact_match"

        def score(self, test_case: TestCase, output: ModelOutput) -> EvalResult:
            match = test_case.reference == output.output
            return EvalResult(
                test_case_id=test_case.id,
                method=self.name,
                score=1.0 if match else 0.0,
                latency_ms=0.0,
            )

    scorer = ExactMatchScorer()
    result = scorer.score(test_case, model_output)
    assert result.score == 1.0
    assert result.method == "exact_match"
