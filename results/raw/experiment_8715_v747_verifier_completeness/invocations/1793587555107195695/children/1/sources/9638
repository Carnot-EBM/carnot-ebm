#!/usr/bin/env python3
"""REQ-REPORT-8069: execute the independent capstone from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v698_capstone import main  # noqa: E402

MODEL_SPECS: list[str] = []

if __name__ == "__main__":
    raise SystemExit(main())
