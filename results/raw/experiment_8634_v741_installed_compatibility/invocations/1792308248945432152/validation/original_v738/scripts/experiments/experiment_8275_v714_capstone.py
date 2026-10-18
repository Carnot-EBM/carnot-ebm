#!/usr/bin/env python3
"""REQ-REPORT-8275: direct imports work from private scratch for cold replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v714_capstone import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
