#!/usr/bin/env python
"""CLI wrapper for Exp 5122 archive .469 / activate .470 aggregation."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.experiment_5122_archive_469_activate_470 import (
    RESULT_RELATIVE_PATH,
    main as cli_main,
    run as run_main,
)


def main(root: Path = REPO_ROOT, *, date: str = "20260701") -> Path:
    """Run the tested Exp 5122 implementation and return the artifact path."""

    output = run_main(root=root, run_date=date)
    return output if output.is_absolute() else root / RESULT_RELATIVE_PATH


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
