#!/usr/bin/env python3
"""Run Exp 4327 ar25 E3 executable-world-model attempt."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.experiment_4327_e3_executable_world_model_ar25 import main


if __name__ == "__main__":
    raise SystemExit(main())
