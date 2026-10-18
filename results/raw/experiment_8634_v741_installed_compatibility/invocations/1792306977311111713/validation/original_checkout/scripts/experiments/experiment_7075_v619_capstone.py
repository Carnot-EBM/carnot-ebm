#!/usr/bin/env python3
"""Run the aggregation-only V619 capstone from REQ-REPORT-7075."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7075_v619_capstone import main


if __name__ == "__main__":
    raise SystemExit(main())
