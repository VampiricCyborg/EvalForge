"""Run-over-run regression detection for eval runs on the same dataset."""

from pydantic import BaseModel

from evalforge.schema import RunReport
from evalforge.storage import Storage

DEFAULT_THRESHOLD = 0.1


class Regression(BaseModel):
    """A single (test case, method) pair whose score dropped materially."""

    test_case_id: str
    method: str
    baseline_score: float
    current_score: float
    # current - baseline, so a regression is always negative.
    delta: float


class ComparisonReport(BaseModel):
    """The result of comparing a current run against a baseline run."""

    baseline_run_id: str
    current_run_id: str
    threshold: float
    regressions: list[Regression]
    regression_count: int
    compared_count: int
    dataset_mismatch: bool
    warnings: list[str]


def _index_scores(report: RunReport) -> dict[tuple[str, str], float]:
    return {
        (result.test_case_id, result.method): result.score
        for result in report.results
    }


class RegressionComparator:
    """Compares two stored runs and flags per-case score drops."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def compare(
        self,
        baseline_run_id: str,
        current_run_id: str,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> ComparisonReport:
        baseline = self.storage.load_run(baseline_run_id)
        current = self.storage.load_run(current_run_id)

        baseline_scores = _index_scores(baseline)
        current_scores = _index_scores(current)
        shared_keys = sorted(baseline_scores.keys() & current_scores.keys())

        warnings: list[str] = []
        dataset_mismatch = baseline.dataset_hash != current.dataset_hash
        if dataset_mismatch:
            warnings.append(
                "dataset hash differs between runs "
                f"(baseline {baseline.dataset_hash!r}, current {current.dataset_hash!r});"
                " comparison may not be meaningful"
            )

        unshared = len(
            baseline_scores.keys() ^ current_scores.keys()  # symmetric difference
        )
        if unshared:
            warnings.append(
                f"{unshared} test case/method pair(s) present in only one run were skipped"
            )

        regressions = []
        for key in shared_keys:
            test_case_id, method = key
            baseline_score = baseline_scores[key]
            current_score = current_scores[key]
            delta = current_score - baseline_score
            if baseline_score - current_score > threshold:
                regressions.append(
                    Regression(
                        test_case_id=test_case_id,
                        method=method,
                        baseline_score=baseline_score,
                        current_score=current_score,
                        delta=delta,
                    )
                )

        return ComparisonReport(
            baseline_run_id=baseline_run_id,
            current_run_id=current_run_id,
            threshold=threshold,
            regressions=regressions,
            regression_count=len(regressions),
            compared_count=len(shared_keys),
            dataset_mismatch=dataset_mismatch,
            warnings=warnings,
        )
