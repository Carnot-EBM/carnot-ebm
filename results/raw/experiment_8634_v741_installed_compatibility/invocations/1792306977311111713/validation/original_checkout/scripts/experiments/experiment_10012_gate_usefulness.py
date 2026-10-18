#!/usr/bin/env python
"""Thin entry point for Experiment 10012 (REQ-ARC-WMTE-10012)."""

from __future__ import annotations

import sys
from pathlib import Path

_PYTHON_DIR = Path(__file__).resolve().parents[2] / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.experiment_10012_gate_usefulness import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
