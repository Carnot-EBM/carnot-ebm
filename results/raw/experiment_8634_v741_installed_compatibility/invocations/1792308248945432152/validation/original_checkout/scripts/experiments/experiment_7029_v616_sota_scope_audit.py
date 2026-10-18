#!/usr/bin/env python3
"""Run the V616 source audit required by REQ-REPORT-7029."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7029_v616_sota_scope_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
