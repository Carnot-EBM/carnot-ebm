#!/usr/bin/env python3
"""CLI entrypoint for Exp 5167 V473 capstone reconciliation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct CLI execution.
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct CLI execution.
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_5167_capstone_v473 import (  # noqa: E402
    DEFAULT_TESTS_RUN,
    AdversarialReporter,
    LevelupLintResult,
    run,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = REPO_ROOT,
    date: str = "20260702",
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    adversarial_reporter: AdversarialReporter | None = None,
    levelup_lint_result: LevelupLintResult | None = None,
) -> Path:
    if argv is not None:
        args = _parse_args(argv)
        root = args.root
        date = args.date
    return run(
        root=root,
        run_date=date,
        duration_s=duration_s,
        tests_run=tests_run if tests_run is not None else DEFAULT_TESTS_RUN,
        adversarial_reporter=adversarial_reporter,
        levelup_lint_result=levelup_lint_result,
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260702", help="Run label date, formatted YYYYMMDD.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root to inspect.")
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover - direct CLI execution.
    main(sys.argv[1:])
