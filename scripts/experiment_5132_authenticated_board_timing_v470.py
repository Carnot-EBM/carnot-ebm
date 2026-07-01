#!/usr/bin/env python3
"""CLI wrapper for Exp 5132 authenticated board timing continuity."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_5132_authenticated_board_timing import main as experiment_main


def main(argv: Sequence[str] | None = None) -> int:
    return experiment_main(argv)


if __name__ == "__main__":  # pragma: no cover - direct script execution.
    raise SystemExit(main())
