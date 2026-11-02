#!/usr/bin/env python
"""CLI wrapper for Exp 5150 archive .471 / activate .472 aggregation."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.experiment_5150_archive_471_activate_472 import (
    RESULT_RELATIVE_PATH,
    main as cli_main,
    run as run_main,
)


def main(root: Path = REPO_ROOT, *, date: str = "20260702") -> Path:
    """Run the tested Exp 5150 implementation and return the artifact path."""

    output = run_main(root=root, run_date=date)
    return output if output.is_absolute() else root / RESULT_RELATIVE_PATH


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
