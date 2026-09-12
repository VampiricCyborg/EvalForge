"""Pydantic models shared across the eval harness."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single test case loaded from a JSON dataset."""

    id: str
    input: str
    reference: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelOutput(BaseModel):
    """The output produced by the model under test for a given test case."""

    test_case_id: str
    output: str


class EvalResult(BaseModel):
    """The score produced by a single scorer for a single test case."""

    test_case_id: str
    method: str
    score: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    raw_output: dict[str, Any] | None = None


class RunReport(BaseModel):
    """The full report for one eval run across a dataset."""

    run_id: str
    timestamp: datetime
    results: list[EvalResult]
    aggregate: dict[str, Any]
    # Identifies the dataset a run was produced from, so run-over-run
    # comparisons can warn when the underlying test cases changed.
    dataset_hash: str | None = None
