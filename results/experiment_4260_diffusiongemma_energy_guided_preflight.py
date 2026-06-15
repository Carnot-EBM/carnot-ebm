#!/usr/bin/env python3
"""CLI wrapper for Exp 4260 DiffusionGemma energy-guided preflight."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.experiment_4260_diffusiongemma_energy_guided_preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
