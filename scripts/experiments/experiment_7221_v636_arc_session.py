#!/usr/bin/env python3
"""Run the Exp7221 V636 cumulative ARC selfparse session."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_7221_v636_arc_session import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
