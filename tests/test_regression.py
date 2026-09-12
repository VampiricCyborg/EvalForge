from datetime import datetime, timezone

import pytest

from evalforge.regression import RegressionComparator
from evalforge.schema import EvalResult, RunReport
from evalforge.storage import Storage


@pytest.fixture
def storage(tmp_path) -> Storage:
    return Storage(tmp_path / "test.db")


@pytest.fixture
def comparator(storage) -> RegressionComparator:
    return RegressionComparator(storage)


def make_report(
    run_id: str,
    scores: dict[tuple[str, str], float],
    dataset_hash: str = "hash-abc",
) -> RunReport:
    """Build a run report from {(test_case_id, method): score}."""
    return RunReport(
        run_id=run_id,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        dataset_hash=dataset_hash,
        aggregate={},
        results=[
            EvalResult(
                test_case_id=case_id,
                method=method,
                score=score,
                latency_ms=1.0,
            )
            for (case_id, method), score in scores.items()
        ],
    )


def test_flags_only_drops_beyond_threshold(storage, comparator):
    baseline = make_report(
        "base",
        {
            ("c1", "exact_match"): 1.0,  # -> 0.4, a 0.6 drop: regression
            ("c2", "exact_match"): 0.5,  # -> 0.45, a 0.05 drop: within threshold
            ("c3", "exact_match"): 0.5,  # -> 0.9, improved
            ("c4", "exact_match"): 0.8,  # -> 0.8, unchanged
        },
    )
    current = make_report(
        "curr",
        {
            ("c1", "exact_match"): 0.4,
            ("c2", "exact_match"): 0.45,
            ("c3", "exact_match"): 0.9,
            ("c4", "exact_match"): 0.8,
        },
    )
    storage.save_run(baseline)
    storage.save_run(current)

    report = comparator.compare("base", "curr", threshold=0.1)

    assert report.regression_count == 1
    assert report.compared_count == 4
    regression = report.regressions[0]
    assert regression.test_case_id == "c1"
    assert regression.method == "exact_match"
    assert regression.baseline_score == 1.0
    assert regression.current_score == 0.4
    assert regression.delta == pytest.approx(-0.6)
    assert report.warnings == []


def test_drop_exactly_at_threshold_is_not_flagged(storage, comparator):
    storage.save_run(make_report("base", {("c1", "exact_match"): 0.5}))
    storage.save_run(make_report("curr", {("c1", "exact_match"): 0.4}))

    report = comparator.compare("base", "curr", threshold=0.1)

    assert report.regression_count == 0


def test_regressions_detected_per_method(storage, comparator):
    baseline = make_report(
        "base",
        {
            ("c1", "exact_match"): 1.0,
            ("c1", "semantic_similarity"): 0.9,
            ("c1", "llm_judge"): 0.8,
        },
    )
    current = make_report(
        "curr",
        {
            ("c1", "exact_match"): 0.0,  # regression
            ("c1", "semantic_similarity"): 0.88,  # noise
            ("c1", "llm_judge"): 0.2,  # regression
        },
    )
    storage.save_run(baseline)
    storage.save_run(current)

    report = comparator.compare("base", "curr")

    flagged = {(r.test_case_id, r.method) for r in report.regressions}
    assert flagged == {("c1", "exact_match"), ("c1", "llm_judge")}
    assert report.compared_count == 3


def test_threshold_is_configurable(storage, comparator):
    storage.save_run(make_report("base", {("c1", "exact_match"): 1.0}))
    storage.save_run(make_report("curr", {("c1", "exact_match"): 0.85}))

    assert comparator.compare("base", "curr", threshold=0.1).regression_count == 1
    assert comparator.compare("base", "curr", threshold=0.2).regression_count == 0


def test_only_shared_test_cases_are_compared(storage, comparator):
    storage.save_run(
        make_report(
            "base",
            {("c1", "exact_match"): 1.0, ("only_in_base", "exact_match"): 1.0},
        )
    )
    storage.save_run(
        make_report(
            "curr",
            {("c1", "exact_match"): 0.1, ("only_in_curr", "exact_match"): 0.0},
        )
    )

    report = comparator.compare("base", "curr")

    assert report.compared_count == 1
    assert report.regression_count == 1
    assert report.regressions[0].test_case_id == "c1"
    assert any("only one run" in w for w in report.warnings)


def test_differing_dataset_hash_warns_but_still_compares(storage, comparator):
    storage.save_run(
        make_report("base", {("c1", "exact_match"): 1.0}, dataset_hash="hash-aaa")
    )
    storage.save_run(
        make_report("curr", {("c1", "exact_match"): 0.2}, dataset_hash="hash-bbb")
    )

    report = comparator.compare("base", "curr")

    assert report.dataset_mismatch is True
    assert any("dataset hash differs" in w for w in report.warnings)
    assert report.regression_count == 1


def test_matching_dataset_hash_does_not_warn(storage, comparator):
    storage.save_run(make_report("base", {("c1", "exact_match"): 1.0}))
    storage.save_run(make_report("curr", {("c1", "exact_match"): 1.0}))

    report = comparator.compare("base", "curr")

    assert report.dataset_mismatch is False
    assert report.warnings == []


def test_no_regressions_when_scores_improve(storage, comparator):
    storage.save_run(
        make_report(
            "base", {("c1", "exact_match"): 0.2, ("c2", "exact_match"): 0.3}
        )
    )
    storage.save_run(
        make_report(
            "curr", {("c1", "exact_match"): 0.9, ("c2", "exact_match"): 1.0}
        )
    )

    report = comparator.compare("base", "curr")

    assert report.regressions == []
    assert report.regression_count == 0
    assert report.compared_count == 2
