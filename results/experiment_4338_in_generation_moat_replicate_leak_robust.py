#!/usr/bin/env python3
"""CLI wrapper for Exp 4338 leak-robust in-generation moat replication."""
# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.experiment_4338_in_generation_moat_replicate_leak_robust import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
