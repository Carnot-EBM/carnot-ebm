#!/usr/bin/env python3
"""Run the Exp7186 matched ARC adapter-withheld transfer pilot."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_7186_v633_arc_withheld_transfer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
