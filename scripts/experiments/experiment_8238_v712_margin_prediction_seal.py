#!/usr/bin/env python3
"""REQ-REPORT-8238: fresh children resolve imports from any private directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.margin_prediction_seal_8238 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
