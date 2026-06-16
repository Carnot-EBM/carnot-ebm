#!/usr/bin/env python3
"""CLI wrapper for Exp 4293 DiffusionGemma partial-state guided run."""
# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.experiment_4293_diffusiongemma_energy_guided_run_partial_state import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
