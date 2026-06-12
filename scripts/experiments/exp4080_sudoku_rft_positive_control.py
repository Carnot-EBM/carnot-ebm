"""CLI entrypoint for Exp 4080 Sudoku verifier-RFT positive control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from carnot.agentic.sudoku_exp4080_rft_positive_control import (
    DEFAULT_SOURCE_RESULT,
    REPO_ROOT,
    RESULT_FILENAME,
    run_experiment,
)


def parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / DEFAULT_SOURCE_RESULT,
        help="Prior live-GPU Sudoku beachhead artifact to aggregate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / RESULT_FILENAME,
        help="Path for the terminal Exp 4080 JSON artifact.",
    )
    return parser.parse_args()


def main() -> int:  # pragma: no cover
    args = parse_args()
    artifact = run_experiment(source_path=args.source, output_path=args.output)
    print(json.dumps(artifact, sort_keys=True))
    return 0 if str(artifact["honest_verdict"]).startswith("complete:") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
