"""Terminal rendering of eval runs and run-over-run comparisons."""

import statistics
from typing import Any

from rich.console import Console
from rich.table import Table

from evalforge.regression import ComparisonReport
from evalforge.schema import EvalResult, RunReport


def build_aggregate(results: list[EvalResult]) -> dict[str, Any]:
    """Summarize a run: mean score per method and high-variance flag count."""
    by_method: dict[str, list[float]] = {}
    for result in results:
        by_method.setdefault(result.method, []).append(result.score)

    return {
        "test_case_count": len({result.test_case_id for result in results}),
        "methods": {
            method: {
                "mean_score": statistics.mean(scores),
                "count": len(scores),
            }
            for method, scores in sorted(by_method.items())
        },
        "high_variance_count": sum(1 for r in results if _is_high_variance(r)),
    }


def _is_high_variance(result: EvalResult) -> bool:
    return bool(result.raw_output and result.raw_output.get("high_variance"))


def _score_style(score: float) -> str:
    if score >= 0.8:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


def render_run_report(report: RunReport, console: Console | None = None) -> None:
    """Print the per-case table and aggregate summary for a run."""
    console = console or Console()

    console.print()
    console.print(f"[bold]Run:[/bold] {report.run_id}")
    console.print(f"[bold]Timestamp:[/bold] {report.timestamp.isoformat()}")
    if report.dataset_hash:
        console.print(f"[bold]Dataset:[/bold] {report.dataset_hash}")

    table = Table(title="Per-test-case results", title_justify="left")
    table.add_column("Test case")
    table.add_column("Method")
    table.add_column("Score", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Flags")

    for result in report.results:
        flags = ""
        if _is_high_variance(result):
            std_dev = result.raw_output.get("std_dev", 0.0)
            flags = f"[magenta]HIGH VARIANCE (sd={std_dev:.3f})[/magenta]"
        table.add_row(
            result.test_case_id,
            result.method,
            f"[{_score_style(result.score)}]{result.score:.3f}[/]",
            f"{result.latency_ms:.1f}",
            flags,
        )
    console.print(table)

    _render_aggregate(report.aggregate, console)


def _render_aggregate(aggregate: dict[str, Any], console: Console) -> None:
    table = Table(title="Aggregate", title_justify="left")
    table.add_column("Method")
    table.add_column("Mean score", justify="right")
    table.add_column("Cases", justify="right")

    for method, stats in aggregate.get("methods", {}).items():
        mean_score = stats["mean_score"]
        table.add_row(
            method,
            f"[{_score_style(mean_score)}]{mean_score:.3f}[/]",
            str(stats["count"]),
        )
    console.print(table)

    console.print(f"Test cases: {aggregate.get('test_case_count', 0)}")
    high_variance_count = aggregate.get("high_variance_count", 0)
    style = "magenta" if high_variance_count else "dim"
    console.print(
        f"[{style}]High-variance judge flags: {high_variance_count}[/{style}]"
        " (reported only - scores are not adjusted)"
    )


def render_comparison(
    comparison: ComparisonReport, console: Console | None = None
) -> None:
    """Print flagged regressions between two runs, plus a summary line."""
    console = console or Console()

    console.print()
    console.print(
        f"[bold]Comparing[/bold] baseline {comparison.baseline_run_id}"
        f" -> current {comparison.current_run_id}"
        f" (threshold {comparison.threshold})"
    )
    for warning in comparison.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    if comparison.regressions:
        table = Table(title="Regressions", title_justify="left")
        table.add_column("Test case")
        table.add_column("Method")
        table.add_column("Baseline", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("Delta", justify="right")

        for regression in comparison.regressions:
            table.add_row(
                f"[red]{regression.test_case_id}[/red]",
                regression.method,
                f"{regression.baseline_score:.3f}",
                f"[red]{regression.current_score:.3f}[/red]",
                f"[bold red]{regression.delta:+.3f}[/bold red]",
            )
        console.print(table)
        console.print(
            f"[bold red]{comparison.regression_count} regression(s)[/bold red]"
            f" out of {comparison.compared_count} compared."
        )
    else:
        console.print(
            f"[green]No regressions[/green] across {comparison.compared_count}"
            " compared test case/method pair(s)."
        )
