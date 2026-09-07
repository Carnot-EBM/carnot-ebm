#!/usr/bin/env python3
"""Run the V625 source and model-state refresh for REQ-REPORT-7122."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7122_v625_sota_ingestion import main


if __name__ == "__main__":
    raise SystemExit(main())
