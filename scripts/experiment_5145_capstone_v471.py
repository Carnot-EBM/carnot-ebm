#!/usr/bin/env python3
"""CLI entrypoint for Exp 5145 V471 capstone aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct CLI execution.
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_5145_capstone_v471 import DEFAULT_TESTS_RUN, run


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = REPO_ROOT,
    date: str = "20260702",
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Run the tested Exp 5145 implementation and return the artifact path."""

    if argv is not None:
        args = _parse_args(argv)
        root = args.root
        date = args.date
    return run(
        root=root,
        run_date=date,
        duration_s=duration_s,
        tests_run=tests_run if tests_run is not None else DEFAULT_TESTS_RUN,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260702", help="Run label date, formatted YYYYMMDD.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root to inspect.")
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
