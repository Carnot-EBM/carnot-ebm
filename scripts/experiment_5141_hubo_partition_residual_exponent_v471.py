#!/usr/bin/env python3
"""CLI entrypoint for Exp 5141 partitioned HUBO/Ising 2D PT telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path

from carnot.experiment_5141_hubo_partition_residual_exponent_v471 import (
    REPO_ROOT,
    main as run_main,
)


def main(
    root: Path = REPO_ROOT,
    *,
    date: str = "20260702",
    duration_s: float | None = None,
    tests_run: list[str] | None = None,
) -> Path:
    """Run the tested Exp 5141 implementation and return the artifact path."""

    return run_main(root=root, date=date, duration_s=duration_s, tests_run=tests_run)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260702", help="Run label date, formatted YYYYMMDD.")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(date=args.date)
