#!/usr/bin/env python3
"""Run Exp7072's fail-closed, claim-grade ARC compaction comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_7072_v619_live_arc_compaction_ab import (  # noqa: E402
    CHECKPOINT_RELATIVE_PATH,
    RESULT_RELATIVE_PATH,
    run,
)


def main(argv: list[str] | None = None) -> int:
    """Parse stable paths, execute the experiment, and print its terminal class."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Execution date as YYYYMMDD")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / RESULT_RELATIVE_PATH,
        help="Terminal result JSON path",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / CHECKPOINT_RELATIVE_PATH,
        help="Atomic per-cell checkpoint path",
    )
    args = parser.parse_args(argv)
    artifact = run(
        REPO_ROOT,
        execution_date=args.date,
        output_path=args.output.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
    )
    print(
        f"Exp7072 {artifact['verdict_class']}: {artifact['honest_verdict']} "
        f"-> {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
