#!/usr/bin/env python3
"""Run Exp6776 window-120 shadow-supervisor evidence accrual."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.experiment_6776_arc_shadow_supervisor_accrual import main


if __name__ == "__main__":
    raise SystemExit(main())
