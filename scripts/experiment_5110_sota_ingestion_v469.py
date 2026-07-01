#!/usr/bin/env python3
"""CLI entrypoint for Exp 5110 V469 source-freshness SOTA ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

from carnot.experiment_5110_sota_ingestion_v469 import REPO_ROOT, main as run_main


def main(
    root: Path = REPO_ROOT,
    *,
    date: str = "20260701",
    duration_s: float | None = None,
    tests_run: list[str] | None = None,
) -> Path:
    """Run the tested Exp 5110 implementation and return the artifact path."""

    return run_main(root=root, date=date, duration_s=duration_s, tests_run=tests_run)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260701", help="Run label date, formatted YYYYMMDD.")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(date=args.date)
