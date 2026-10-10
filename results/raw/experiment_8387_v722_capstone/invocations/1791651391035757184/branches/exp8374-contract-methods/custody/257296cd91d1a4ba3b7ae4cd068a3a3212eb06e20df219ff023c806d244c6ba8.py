#!/usr/bin/env python3
"""REQ-REPORT-8374: the thin CLI runs from private working directories too."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v722_contract_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
