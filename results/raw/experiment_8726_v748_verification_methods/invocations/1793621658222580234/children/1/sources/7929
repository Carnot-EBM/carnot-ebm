#!/usr/bin/env python3
"""CLI entrypoint for Exp 5133 .470 capstone aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path

from carnot.experiment_5133_capstone_v470 import (
    DEFAULT_TESTS_RUN,
    REPO_ROOT,
    AdversarialReporter,
    run,
)


def main(
    *,
    root: Path = REPO_ROOT,
    date: str = "20260701",
    duration_s: float | None = None,
    tests_run: list[str] | None = None,
    adversarial_reporter: AdversarialReporter | None = None,
) -> Path:
    """Run the tested Exp 5133 implementation and return the artifact path."""

    kwargs = {}
    if adversarial_reporter is not None:
        kwargs["adversarial_reporter"] = adversarial_reporter
    return run(
        root=root,
        run_date=date,
        duration_s=duration_s,
        tests_run=tests_run if tests_run is not None else DEFAULT_TESTS_RUN,
        **kwargs,
    )


def _parse_args() -> argparse.Namespace:  # pragma: no cover - thin CLI parser.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260701", help="Run label date, formatted YYYYMMDD.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root to inspect.")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - direct CLI execution.
    args = _parse_args()
    main(root=args.root, date=args.date)
