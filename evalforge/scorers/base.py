"""Abstract interface that every scoring method must implement."""

from abc import ABC, abstractmethod

from evalforge.schema import EvalResult, ModelOutput, TestCase


class Scorer(ABC):
    """A scoring method that grades a model output against a test case."""

    name: str

    @abstractmethod
    def score(self, test_case: TestCase, output: ModelOutput) -> EvalResult:
        """Score a single model output against its test case."""
        raise NotImplementedError
