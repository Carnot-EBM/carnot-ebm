#!/usr/bin/env python3
"""CLI entrypoint for Exp 5219 V477 capstone reconciliation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct CLI execution.
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct CLI execution.
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_5219_capstone_v477 import (  # noqa: E402
    DEFAULT_VALIDATION_COMMANDS,
    AdversarialReporter,
    run,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    root: Path = REPO_ROOT,
    date: str = "20260704",
    duration_s: float | None = None,
    validation_commands_run: Sequence[Mapping[str, Any]] | None = None,
    adversarial_reporter: AdversarialReporter | None = None,
    conductor_untouched: bool | None = None,
    docs_reconciled: bool = False,
) -> Path:
    if argv is not None:
        args = _parse_args(argv)
        root = args.root
        date = args.date
    return run(
        root=root,
        run_date=date,
        duration_s=duration_s,
        validation_commands_run=(
            validation_commands_run
            if validation_commands_run is not None
            else DEFAULT_VALIDATION_COMMANDS
        ),
        adversarial_reporter=adversarial_reporter,
        conductor_untouched=conductor_untouched,
        docs_reconciled=docs_reconciled,
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260704", help="Run label date, formatted YYYYMMDD.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root to inspect.")
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover - direct CLI execution.
    main(sys.argv[1:])
