"""Export two stored runs plus their comparison into a single JSON file.

Feeds the static showcase site in docs/ with real data. Run the two evals
through the CLI first, then point this at their run ids:

    uv run python scripts/export_showcase_data.py \
        --baseline showcase-v1 --current showcase-v2
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evalforge.cli import load_model_outputs, load_test_cases
from evalforge.regression import RegressionComparator
from evalforge.schema import RunReport
from evalforge.storage import Storage

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "data" / "example-run.json"


def serialize_run(report: RunReport, outputs_path: Path) -> dict:
    outputs = load_model_outputs(outputs_path)
    return {
        "run_id": report.run_id,
        "timestamp": report.timestamp.isoformat(),
        "dataset_hash": report.dataset_hash,
        "outputs_file": outputs_path.name,
        "aggregate": report.aggregate,
        "model_outputs": {
            case_id: output.output for case_id, output in outputs.items()
        },
        "results": [result.model_dump() for result in report.results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Baseline run id")
    parser.add_argument("--current", required=True, help="Current run id")
    parser.add_argument("--db", default=str(REPO_ROOT / "evalforge.db"))
    parser.add_argument("--dataset", default=str(REPO_ROOT / "examples" / "tests.json"))
    parser.add_argument(
        "--baseline-outputs", default=str(REPO_ROOT / "examples" / "outputs.json")
    )
    parser.add_argument(
        "--current-outputs", default=str(REPO_ROOT / "examples" / "outputs_v2.json")
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    storage = Storage(args.db)
    baseline = storage.load_run(args.baseline)
    current = storage.load_run(args.current)
    comparison = RegressionComparator(storage).compare(
        args.baseline, args.current, args.threshold
    )

    test_cases = load_test_cases(Path(args.dataset))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "file": Path(args.dataset).name,
            "hash": baseline.dataset_hash,
            "test_cases": [case.model_dump() for case in test_cases],
        },
        "runs": {
            "baseline": serialize_run(baseline, Path(args.baseline_outputs)),
            "current": serialize_run(current, Path(args.current_outputs)),
        },
        "comparison": comparison.model_dump(),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_path} ({output_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
