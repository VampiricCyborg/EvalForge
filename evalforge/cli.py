"""CLI entrypoint for EvalForge."""

import argparse
import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from evalforge import __version__
from evalforge.regression import DEFAULT_THRESHOLD, RegressionComparator
from evalforge.reporting import build_aggregate, render_comparison, render_run_report
from evalforge.schema import EvalResult, ModelOutput, RunReport, TestCase
from evalforge.scorers import SCORERS
from evalforge.scorers.llm_judge import JudgeError
from evalforge.storage import DEFAULT_DB_PATH, RunNotFoundError, Storage

EXIT_OK = 0
EXIT_REGRESSIONS_FOUND = 1
EXIT_ERROR = 2


class CliError(Exception):
    """A user-facing error that should exit cleanly rather than traceback."""


def _load_json_array(path: Path) -> list[dict]:
    if not path.exists():
        raise CliError(f"file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise CliError(f"{path} must contain a JSON array")
    return data


def load_test_cases(path: Path) -> list[TestCase]:
    try:
        return [TestCase.model_validate(item) for item in _load_json_array(path)]
    except ValidationError as exc:
        raise CliError(f"invalid test case in {path}: {exc}") from exc


def load_model_outputs(path: Path) -> dict[str, ModelOutput]:
    try:
        outputs = [ModelOutput.model_validate(item) for item in _load_json_array(path)]
    except ValidationError as exc:
        raise CliError(f"invalid model output in {path}: {exc}") from exc
    return {output.test_case_id: output for output in outputs}


def compute_dataset_hash(test_cases: list[TestCase]) -> str:
    """Hash the dataset's content so runs can be matched to the cases they used."""
    canonical = json.dumps(
        [case.model_dump() for case in sorted(test_cases, key=lambda c: c.id)],
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def resolve_scorers(names: str) -> list[str]:
    requested = [name.strip() for name in names.split(",") if name.strip()]
    if not requested:
        raise CliError("no scorers requested")
    unknown = [name for name in requested if name not in SCORERS]
    if unknown:
        raise CliError(
            f"unknown scorer(s): {', '.join(unknown)}."
            f" Available: {', '.join(sorted(SCORERS))}"
        )
    return requested


def generate_run_id(timestamp: datetime) -> str:
    return f"run-{timestamp.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def cmd_run(args: argparse.Namespace, console: Console) -> int:
    test_cases = load_test_cases(Path(args.dataset))
    outputs = load_model_outputs(Path(args.outputs))
    scorer_names = resolve_scorers(args.scorers)

    missing = [case.id for case in test_cases if case.id not in outputs]
    if missing:
        console.print(
            f"[yellow]Warning:[/yellow] no model output for {len(missing)} test"
            f" case(s), skipping: {', '.join(missing)}"
        )
    scored_cases = [case for case in test_cases if case.id in outputs]
    if not scored_cases:
        raise CliError("no test cases have a matching model output")

    try:
        scorers = [SCORERS[name]() for name in scorer_names]
    except ValueError as exc:
        raise CliError(str(exc)) from exc

    results: list[EvalResult] = []
    for case in scored_cases:
        for scorer in scorers:
            results.append(scorer.score(case, outputs[case.id]))

    timestamp = datetime.now(timezone.utc)
    report = RunReport(
        run_id=args.run_id or generate_run_id(timestamp),
        timestamp=timestamp,
        results=results,
        aggregate=build_aggregate(results),
        dataset_hash=compute_dataset_hash(test_cases),
    )

    try:
        Storage(args.db).save_run(report)
    except sqlite3.IntegrityError as exc:
        raise CliError(
            f"could not save run {report.run_id!r}: {exc}."
            " Run ids must be unique - omit --run-id to generate one."
        ) from exc
    render_run_report(report, console)
    console.print(f"\nSaved run [bold]{report.run_id}[/bold] to {args.db}")
    return EXIT_OK


def cmd_compare(args: argparse.Namespace, console: Console) -> int:
    comparator = RegressionComparator(Storage(args.db))
    comparison = comparator.compare(args.baseline, args.current, args.threshold)
    render_comparison(comparison, console)
    return EXIT_REGRESSIONS_FOUND if comparison.regressions else EXIT_OK


def cmd_list(args: argparse.Namespace, console: Console) -> int:
    runs = Storage(args.db).list_runs()
    if not runs:
        console.print(f"No runs stored in {args.db}.")
        return EXIT_OK

    table = Table(title="Stored runs", title_justify="left")
    table.add_column("Run ID")
    table.add_column("Timestamp")
    for run_id, timestamp in runs:
        table.add_row(run_id, timestamp.isoformat())
    console.print(table)
    return EXIT_OK


def cmd_report(args: argparse.Namespace, console: Console) -> int:
    render_run_report(Storage(args.db).load_run(args.run_id), console)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalforge", description="LLM eval harness with regression detection"
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_db_arg(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--db", default=DEFAULT_DB_PATH, help="SQLite database path"
        )

    run_parser = subparsers.add_parser("run", help="Score model outputs and store a run")
    run_parser.add_argument("--dataset", required=True, help="JSON file of test cases")
    run_parser.add_argument(
        "--outputs", required=True, help="JSON file of model outputs"
    )
    run_parser.add_argument(
        "--scorers",
        default="exact_match",
        help=f"Comma-separated scorers. Available: {', '.join(sorted(SCORERS))}",
    )
    run_parser.add_argument("--run-id", default=None, help="Defaults to a generated id")
    add_db_arg(run_parser)
    run_parser.set_defaults(handler=cmd_run)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two runs (exits 1 when regressions are found)",
    )
    compare_parser.add_argument("--baseline", required=True, help="Baseline run id")
    compare_parser.add_argument("--current", required=True, help="Current run id")
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Score drop that counts as a regression",
    )
    add_db_arg(compare_parser)
    compare_parser.set_defaults(handler=cmd_compare)

    list_parser = subparsers.add_parser("list", help="List stored runs")
    add_db_arg(list_parser)
    list_parser.set_defaults(handler=cmd_list)

    report_parser = subparsers.add_parser("report", help="Re-print a stored run")
    report_parser.add_argument("--run-id", required=True)
    add_db_arg(report_parser)
    report_parser.set_defaults(handler=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        return args.handler(args, console)
    except (CliError, RunNotFoundError, JudgeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
