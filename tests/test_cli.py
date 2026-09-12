import json
import subprocess
import sys
from pathlib import Path

import pytest

from evalforge.cli import EXIT_ERROR, EXIT_OK, EXIT_REGRESSIONS_FOUND, main

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def db(tmp_path) -> str:
    return str(tmp_path / "cli.db")


@pytest.fixture
def dataset(tmp_path) -> str:
    cases = [
        {"id": "c1", "input": "What is 2+2?", "reference": "4"},
        {"id": "c2", "input": "Capital of France?", "reference": "Paris"},
        {"id": "c3", "input": "Largest ocean?", "reference": "Pacific"},
    ]
    path = tmp_path / "tests.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return str(path)


def write_outputs(tmp_path, name: str, outputs: dict[str, str]) -> str:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            [{"test_case_id": k, "output": v} for k, v in outputs.items()]
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def good_outputs(tmp_path) -> str:
    return write_outputs(
        tmp_path, "good.json", {"c1": "4", "c2": "Paris", "c3": "Pacific"}
    )


@pytest.fixture
def bad_outputs(tmp_path) -> str:
    # c2 and c3 regress; c1 stays correct.
    return write_outputs(
        tmp_path, "bad.json", {"c1": "4", "c2": "Berlin", "c3": "Atlantic"}
    )


def run_cli(*argv: str) -> int:
    return main(list(argv))


class TestRunCommand:
    def test_run_scores_and_saves(self, capsys, db, dataset, good_outputs):
        exit_code = run_cli(
            "run",
            "--dataset", dataset,
            "--outputs", good_outputs,
            "--scorers", "exact_match",
            "--run-id", "r1",
            "--db", db,
        )
        out = capsys.readouterr().out

        assert exit_code == EXIT_OK
        assert "r1" in out
        assert "exact_match" in out
        assert "1.000" in out
        assert "Aggregate" in out
        assert "High-variance judge flags: 0" in out

    def test_run_generates_run_id_when_omitted(self, capsys, db, dataset, good_outputs):
        exit_code = run_cli(
            "run",
            "--dataset", dataset,
            "--outputs", good_outputs,
            "--scorers", "exact_match",
            "--db", db,
        )
        assert exit_code == EXIT_OK
        assert "run-" in capsys.readouterr().out

    def test_run_with_multiple_scorers(self, capsys, db, dataset, good_outputs):
        exit_code = run_cli(
            "run",
            "--dataset", dataset,
            "--outputs", good_outputs,
            "--scorers", "exact_match,semantic",
            "--run-id", "r1",
            "--db", db,
        )
        out = capsys.readouterr().out

        assert exit_code == EXIT_OK
        assert "exact_match" in out
        assert "semantic" in out

    def test_unknown_scorer_errors(self, capsys, db, dataset, good_outputs):
        exit_code = run_cli(
            "run",
            "--dataset", dataset,
            "--outputs", good_outputs,
            "--scorers", "not_a_scorer",
            "--db", db,
        )
        out = capsys.readouterr().out

        assert exit_code == EXIT_ERROR
        assert "unknown scorer" in out.lower()

    def test_missing_dataset_file_errors(self, capsys, db, good_outputs):
        exit_code = run_cli(
            "run",
            "--dataset", "does_not_exist.json",
            "--outputs", good_outputs,
            "--db", db,
        )
        assert exit_code == EXIT_ERROR
        assert "not found" in capsys.readouterr().out.lower()

    def test_malformed_json_errors(self, capsys, tmp_path, db, good_outputs):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")

        exit_code = run_cli(
            "run", "--dataset", str(bad), "--outputs", good_outputs, "--db", db
        )
        assert exit_code == EXIT_ERROR
        assert "not valid json" in capsys.readouterr().out.lower()

    def test_missing_output_for_case_warns_and_continues(
        self, capsys, tmp_path, db, dataset
    ):
        partial = write_outputs(tmp_path, "partial.json", {"c1": "4"})

        exit_code = run_cli(
            "run",
            "--dataset", dataset,
            "--outputs", partial,
            "--scorers", "exact_match",
            "--run-id", "r1",
            "--db", db,
        )
        out = capsys.readouterr().out

        assert exit_code == EXIT_OK
        assert "Warning" in out
        assert "c2" in out

    def test_no_matching_outputs_errors(self, capsys, tmp_path, db, dataset):
        unrelated = write_outputs(tmp_path, "unrelated.json", {"zzz": "nope"})

        exit_code = run_cli(
            "run", "--dataset", dataset, "--outputs", unrelated, "--db", db
        )
        assert exit_code == EXIT_ERROR
        assert "no test cases" in capsys.readouterr().out.lower()

    def test_duplicate_run_id_errors_cleanly(self, capsys, db, dataset, good_outputs):
        argv = [
            "run",
            "--dataset", dataset,
            "--outputs", good_outputs,
            "--scorers", "exact_match",
            "--run-id", "r1",
            "--db", db,
        ]
        assert main(argv) == EXIT_OK
        capsys.readouterr()

        assert main(argv) == EXIT_ERROR
        assert "unique" in capsys.readouterr().out.lower()

    def test_llm_judge_without_api_key_errors_cleanly(
        self, capsys, monkeypatch, db, dataset, good_outputs
    ):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        exit_code = run_cli(
            "run",
            "--dataset", dataset,
            "--outputs", good_outputs,
            "--scorers", "llm_judge",
            "--db", db,
        )
        assert exit_code == EXIT_ERROR
        assert "GROQ_API_KEY" in capsys.readouterr().out


class TestCompareCommand:
    @pytest.fixture
    def two_runs(self, db, dataset, good_outputs, bad_outputs, capsys):
        run_cli(
            "run", "--dataset", dataset, "--outputs", good_outputs,
            "--scorers", "exact_match", "--run-id", "base", "--db", db,
        )
        run_cli(
            "run", "--dataset", dataset, "--outputs", bad_outputs,
            "--scorers", "exact_match", "--run-id", "curr", "--db", db,
        )
        capsys.readouterr()  # discard run output
        return db

    def test_regressions_are_flagged_and_exit_nonzero(self, capsys, two_runs):
        exit_code = run_cli(
            "compare", "--baseline", "base", "--current", "curr", "--db", two_runs
        )
        out = capsys.readouterr().out

        assert exit_code == EXIT_REGRESSIONS_FOUND
        assert "2 regression(s)" in out
        assert "c2" in out
        assert "c3" in out
        assert "-1.000" in out
        # c1 did not regress and must not appear in the regression table.
        assert "c1" not in out

    def test_no_regressions_exits_zero(self, capsys, two_runs):
        exit_code = run_cli(
            "compare", "--baseline", "base", "--current", "base", "--db", two_runs
        )
        out = capsys.readouterr().out

        assert exit_code == EXIT_OK
        assert "No regressions" in out

    def test_threshold_is_respected(self, capsys, two_runs):
        exit_code = run_cli(
            "compare", "--baseline", "base", "--current", "curr",
            "--threshold", "1.5", "--db", two_runs,
        )
        assert exit_code == EXIT_OK
        assert "No regressions" in capsys.readouterr().out

    def test_unknown_run_id_errors(self, capsys, two_runs):
        exit_code = run_cli(
            "compare", "--baseline", "ghost", "--current", "curr", "--db", two_runs
        )
        assert exit_code == EXIT_ERROR
        assert "no run with run_id" in capsys.readouterr().out.lower()

    def test_dataset_mismatch_warns(self, capsys, tmp_path, db, dataset, good_outputs):
        other_dataset = tmp_path / "other.json"
        other_dataset.write_text(
            json.dumps([{"id": "c1", "input": "different", "reference": "4"}]),
            encoding="utf-8",
        )
        run_cli(
            "run", "--dataset", dataset, "--outputs", good_outputs,
            "--scorers", "exact_match", "--run-id", "base", "--db", db,
        )
        run_cli(
            "run", "--dataset", str(other_dataset), "--outputs", good_outputs,
            "--scorers", "exact_match", "--run-id", "curr", "--db", db,
        )
        capsys.readouterr()

        exit_code = run_cli(
            "compare", "--baseline", "base", "--current", "curr", "--db", db
        )
        out = capsys.readouterr().out

        assert exit_code == EXIT_OK
        assert "dataset hash differs" in out


class TestListCommand:
    def test_empty_database(self, capsys, db):
        assert run_cli("list", "--db", db) == EXIT_OK
        assert "No runs stored" in capsys.readouterr().out

    def test_lists_saved_runs(self, capsys, db, dataset, good_outputs):
        run_cli(
            "run", "--dataset", dataset, "--outputs", good_outputs,
            "--scorers", "exact_match", "--run-id", "r1", "--db", db,
        )
        capsys.readouterr()

        assert run_cli("list", "--db", db) == EXIT_OK
        out = capsys.readouterr().out
        assert "r1" in out
        assert "Stored runs" in out


class TestReportCommand:
    def test_reprints_stored_run(self, capsys, db, dataset, good_outputs):
        run_cli(
            "run", "--dataset", dataset, "--outputs", good_outputs,
            "--scorers", "exact_match", "--run-id", "r1", "--db", db,
        )
        capsys.readouterr()

        assert run_cli("report", "--run-id", "r1", "--db", db) == EXIT_OK
        out = capsys.readouterr().out
        assert "r1" in out
        assert "Per-test-case results" in out
        assert "Aggregate" in out
        assert "exact_match" in out

    def test_unknown_run_id_errors(self, capsys, db):
        assert run_cli("report", "--run-id", "ghost", "--db", db) == EXIT_ERROR
        assert "no run with run_id" in capsys.readouterr().out.lower()


class TestSubprocessEntrypoint:
    """End-to-end check that the CLI runs as a real process against examples/."""

    def test_examples_run_and_compare_via_subprocess(self, tmp_path):
        db = str(tmp_path / "e2e.db")
        common = [sys.executable, "-m", "evalforge.cli"]

        for run_id, outputs in [
            ("v1", EXAMPLES / "outputs.json"),
            ("v2", EXAMPLES / "outputs_v2.json"),
        ]:
            result = subprocess.run(
                common + [
                    "run",
                    "--dataset", str(EXAMPLES / "tests.json"),
                    "--outputs", str(outputs),
                    "--scorers", "exact_match",
                    "--run-id", run_id,
                    "--db", db,
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == EXIT_OK, result.stderr

        compare = subprocess.run(
            common + ["compare", "--baseline", "v1", "--current", "v2", "--db", db],
            capture_output=True,
            text=True,
        )
        assert compare.returncode == EXIT_REGRESSIONS_FOUND, compare.stderr
        assert "capital-france" in compare.stdout
