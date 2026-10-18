#!/usr/bin/env python3
"""CLI entrypoint for Exp 5142 TACO/CSP harm root-cause scaling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_5142_taco_harm_rootcause_scale_v471 import (  # noqa: E402
    RUN_DATE,
    main as run_main,
)


def main(
    root: Path = REPO_ROOT,
    *,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: list[str] | None = None,
) -> Path:
    """Run the tested Exp 5142 implementation and return the artifact path."""

    return run_main(root=root, date=date, duration_s=duration_s, tests_run=tests_run)


def _parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE, help="Run label date, formatted YYYYMMDD.")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(date=args.date)
