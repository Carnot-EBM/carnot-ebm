#!/usr/bin/env python3
"""Run the V623 execution-time source ingestion required by REQ-REPORT-7098."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7098_v623_sota_ingestion import main


if __name__ == "__main__":
    raise SystemExit(main())
