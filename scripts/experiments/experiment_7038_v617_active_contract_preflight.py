#!/usr/bin/env python3
"""Run the advisory V617 active contract preflight from REQ-REPORT-7038."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_7038_v617_active_contract_preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
