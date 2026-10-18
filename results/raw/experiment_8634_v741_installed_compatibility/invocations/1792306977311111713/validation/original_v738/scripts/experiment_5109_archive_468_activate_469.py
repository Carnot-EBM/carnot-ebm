#!/usr/bin/env python3
"""CLI entrypoint for Exp 5109 archive .468 / activate .469 aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

from carnot.reporting.archive_468_activate_469_5109 import (
    REPO_ROOT,
    CommandResult,
    main as run_main,
)


def main(
    root: Path = REPO_ROOT,
    *,
    date: str = "20260701",
    adversarial_result: CommandResult | None = None,
) -> Path:
    """Run the tested Exp 5109 implementation and return the artifact path."""

    return run_main(root=root, date=date, adversarial_result=adversarial_result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260701", help="Run label date, formatted YYYYMMDD.")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(date=args.date)
