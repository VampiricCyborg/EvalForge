from datetime import datetime, timezone

import pytest

from evalforge.schema import EvalResult, ModelOutput, TestCase


@pytest.fixture
def test_case() -> TestCase:
    return TestCase(
        id="case-1",
        input="What is the capital of France?",
        reference="Paris",
        metadata={"category": "geography"},
    )


@pytest.fixture
def model_output() -> ModelOutput:
    return ModelOutput(test_case_id="case-1", output="Paris")


@pytest.fixture
def eval_result() -> EvalResult:
    return EvalResult(
        test_case_id="case-1",
        method="exact_match",
        score=1.0,
        latency_ms=42.5,
        raw_output={"reasoning": "matches reference exactly"},
    )


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)
