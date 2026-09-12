import pytest
from pydantic import ValidationError

from evalforge.schema import EvalResult, ModelOutput, RunReport, TestCase


class TestTestCase:
    def test_valid_minimal(self):
        case = TestCase(id="c1", input="hello")
        assert case.id == "c1"
        assert case.reference is None
        assert case.metadata == {}

    def test_valid_full(self, test_case):
        assert test_case.reference == "Paris"
        assert test_case.metadata["category"] == "geography"

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            TestCase(input="hello")  # missing id

    def test_wrong_type_rejected(self):
        with pytest.raises(ValidationError):
            TestCase(id="c1", input="hello", metadata="not-a-dict")


class TestModelOutput:
    def test_valid(self, model_output):
        assert model_output.test_case_id == "case-1"
        assert model_output.output == "Paris"

    def test_missing_field_rejected(self):
        with pytest.raises(ValidationError):
            ModelOutput(output="Paris")  # missing test_case_id


class TestEvalResult:
    def test_valid(self, eval_result):
        assert eval_result.score == 1.0
        assert eval_result.raw_output["reasoning"]

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            EvalResult(
                test_case_id="case-1",
                method="exact_match",
                score=1.5,
                latency_ms=10.0,
            )

    def test_negative_score_rejected(self):
        with pytest.raises(ValidationError):
            EvalResult(
                test_case_id="case-1",
                method="exact_match",
                score=-0.1,
                latency_ms=10.0,
            )

    def test_negative_latency_rejected(self):
        with pytest.raises(ValidationError):
            EvalResult(
                test_case_id="case-1",
                method="exact_match",
                score=0.5,
                latency_ms=-1.0,
            )

    def test_raw_output_defaults_to_none(self):
        result = EvalResult(
            test_case_id="case-1", method="exact_match", score=0.5, latency_ms=10.0
        )
        assert result.raw_output is None


class TestRunReport:
    def test_valid(self, eval_result, now):
        report = RunReport(
            run_id="run-1",
            timestamp=now,
            results=[eval_result],
            aggregate={"mean_score": 1.0},
        )
        assert report.run_id == "run-1"
        assert len(report.results) == 1
        assert report.aggregate["mean_score"] == 1.0

    def test_missing_aggregate_rejected(self, eval_result, now):
        with pytest.raises(ValidationError):
            RunReport(run_id="run-1", timestamp=now, results=[eval_result])

    def test_invalid_results_type_rejected(self, now):
        with pytest.raises(ValidationError):
            RunReport(
                run_id="run-1",
                timestamp=now,
                results="not-a-list",
                aggregate={},
            )
