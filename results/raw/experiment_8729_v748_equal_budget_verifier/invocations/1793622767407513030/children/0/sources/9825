#!/usr/bin/env python3
"""REQ-REPORT-8304: make fresh-process replay work from outside the checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v717_contract_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
