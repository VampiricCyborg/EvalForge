import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from evalforge.schema import EvalResult, RunReport
from evalforge.storage import RunNotFoundError, Storage


@pytest.fixture
def storage(tmp_path) -> Storage:
    return Storage(tmp_path / "test.db")


def make_report(run_id: str, scores: dict[str, float], **kwargs) -> RunReport:
    return RunReport(
        run_id=run_id,
        timestamp=kwargs.get("timestamp", datetime(2026, 1, 1, tzinfo=timezone.utc)),
        dataset_hash=kwargs.get("dataset_hash", "hash-abc"),
        aggregate=kwargs.get("aggregate", {"mean_score": sum(scores.values())}),
        results=[
            EvalResult(
                test_case_id=case_id,
                method=kwargs.get("method", "exact_match"),
                score=score,
                latency_ms=1.5,
                raw_output=kwargs.get("raw_output", {"note": "n/a"}),
            )
            for case_id, score in scores.items()
        ],
    )


def test_save_and_load_round_trip(storage):
    report = make_report("run-1", {"c1": 1.0, "c2": 0.5})
    storage.save_run(report)

    loaded = storage.load_run("run-1")

    assert loaded.run_id == report.run_id
    assert loaded.timestamp == report.timestamp
    assert loaded.dataset_hash == "hash-abc"
    assert loaded.aggregate == report.aggregate
    assert len(loaded.results) == 2
    assert loaded.results[0].test_case_id == "c1"
    assert loaded.results[0].score == 1.0
    assert loaded.results[0].raw_output == {"note": "n/a"}


def test_round_trip_preserves_nested_raw_output(storage):
    report = make_report(
        "run-1",
        {"c1": 0.6},
        raw_output={"scores": [0.5, 0.7], "high_variance": True, "std_dev": 0.14},
    )
    storage.save_run(report)

    loaded = storage.load_run("run-1")

    assert loaded.results[0].raw_output["scores"] == [0.5, 0.7]
    assert loaded.results[0].raw_output["high_variance"] is True


def test_null_raw_output_round_trips_as_none(storage):
    report = make_report("run-1", {"c1": 1.0}, raw_output=None)
    storage.save_run(report)

    assert storage.load_run("run-1").results[0].raw_output is None


def test_load_missing_run_raises(storage):
    with pytest.raises(RunNotFoundError):
        storage.load_run("nope")


def test_list_runs_returns_newest_first(storage):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    storage.save_run(make_report("run-old", {"c1": 1.0}, timestamp=base))
    storage.save_run(
        make_report("run-new", {"c1": 1.0}, timestamp=base + timedelta(days=1))
    )

    runs = storage.list_runs()

    assert [run_id for run_id, _ in runs] == ["run-new", "run-old"]
    assert runs[0][1] == base + timedelta(days=1)


def test_list_runs_empty_database(storage):
    assert storage.list_runs() == []


def test_duplicate_run_id_rejected(storage):
    storage.save_run(make_report("run-1", {"c1": 1.0}))
    with pytest.raises(sqlite3.IntegrityError):
        storage.save_run(make_report("run-1", {"c1": 0.0}))


def test_storage_persists_across_instances(tmp_path):
    db_path = tmp_path / "persist.db"
    Storage(db_path).save_run(make_report("run-1", {"c1": 1.0}))

    assert Storage(db_path).load_run("run-1").results[0].score == 1.0


def test_results_are_linked_by_foreign_key(storage):
    storage.save_run(make_report("run-1", {"c1": 1.0}))

    with sqlite3.connect(storage.db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO results"
                " (run_id, test_case_id, method, score, latency_ms, raw_output)"
                " VALUES ('ghost-run', 'c1', 'exact_match', 1.0, 1.0, NULL)"
            )
