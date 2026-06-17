"""Run Exp 4329 tr87/ft09 E3 executable-world-model attempt.

Spec refs: REQ-PHASE4-076, SCENARIO-PHASE4-076.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.experiment_4329_e3_executable_world_model_tr87_ft09 import main


if __name__ == "__main__":
    raise SystemExit(main())
