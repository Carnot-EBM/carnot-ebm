#!/usr/bin/env python3
"""Run Exp4341 sc25 E3 offline reproduction."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.experiment_4341_e3_sc25_reproduction import main


if __name__ == "__main__":
    raise SystemExit(main())
