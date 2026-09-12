"""SQLite persistence for eval runs."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from evalforge.schema import EvalResult, RunReport

DEFAULT_DB_PATH = "evalforge.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    timestamp    TEXT NOT NULL,
    dataset_hash TEXT,
    aggregate    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    test_case_id TEXT NOT NULL,
    method       TEXT NOT NULL,
    score        REAL NOT NULL,
    latency_ms   REAL NOT NULL,
    raw_output   TEXT,
    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_results_run_id ON results (run_id);
"""


class RunNotFoundError(LookupError):
    """Raised when a run_id is not present in the database."""


class Storage:
    """Stores and retrieves eval runs in a SQLite database."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        with closing(self._connect()) as conn, conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save_run(self, report: RunReport) -> None:
        """Persist a full run and all of its results in one transaction."""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO runs (run_id, timestamp, dataset_hash, aggregate)"
                " VALUES (?, ?, ?, ?)",
                (
                    report.run_id,
                    report.timestamp.isoformat(),
                    report.dataset_hash,
                    json.dumps(report.aggregate),
                ),
            )
            conn.executemany(
                "INSERT INTO results"
                " (run_id, test_case_id, method, score, latency_ms, raw_output)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        report.run_id,
                        result.test_case_id,
                        result.method,
                        result.score,
                        result.latency_ms,
                        json.dumps(result.raw_output)
                        if result.raw_output is not None
                        else None,
                    )
                    for result in report.results
                ],
            )

    def load_run(self, run_id: str) -> RunReport:
        """Load a previously saved run by id."""
        with closing(self._connect()) as conn:
            run_row = conn.execute(
                "SELECT run_id, timestamp, dataset_hash, aggregate"
                " FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_row is None:
                raise RunNotFoundError(f"no run with run_id {run_id!r}")

            result_rows = conn.execute(
                "SELECT test_case_id, method, score, latency_ms, raw_output"
                " FROM results WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()

        return RunReport(
            run_id=run_row["run_id"],
            timestamp=datetime.fromisoformat(run_row["timestamp"]),
            dataset_hash=run_row["dataset_hash"],
            aggregate=json.loads(run_row["aggregate"]),
            results=[
                EvalResult(
                    test_case_id=row["test_case_id"],
                    method=row["method"],
                    score=row["score"],
                    latency_ms=row["latency_ms"],
                    raw_output=json.loads(row["raw_output"])
                    if row["raw_output"] is not None
                    else None,
                )
                for row in result_rows
            ],
        )

    def list_runs(self) -> list[tuple[str, datetime]]:
        """List stored runs as (run_id, timestamp), newest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT run_id, timestamp FROM runs ORDER BY timestamp DESC, run_id"
            ).fetchall()
        return [
            (row["run_id"], datetime.fromisoformat(row["timestamp"])) for row in rows
        ]
