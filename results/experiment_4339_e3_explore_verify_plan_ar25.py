#!/usr/bin/env python3
"""Run Exp 4339 ar25 E3 explore-verify-plan refinement."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.experiment_4339_e3_explore_verify_plan_ar25 import main


if __name__ == "__main__":
    raise SystemExit(main())
