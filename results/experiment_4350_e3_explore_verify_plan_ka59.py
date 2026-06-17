#!/usr/bin/env python3
"""Run Exp 4350 ka59 E3 explore-verify-plan continuation."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.experiment_4350_e3_explore_verify_plan_ka59 import main


if __name__ == "__main__":
    raise SystemExit(main())
