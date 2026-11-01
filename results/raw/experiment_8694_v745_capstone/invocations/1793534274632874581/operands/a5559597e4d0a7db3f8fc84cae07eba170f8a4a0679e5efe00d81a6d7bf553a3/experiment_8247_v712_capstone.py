#!/usr/bin/env python3
"""REQ-REPORT-8247: direct imports work from private directories without PYTHONPATH."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v712_capstone import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
