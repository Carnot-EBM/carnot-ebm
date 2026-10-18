#!/usr/bin/env python3
"""Run the advisory V614 preflight required by REQ-REPORT-7009."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7009_v614_source_contract_preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
