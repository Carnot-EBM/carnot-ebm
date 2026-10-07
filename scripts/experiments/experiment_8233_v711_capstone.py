#!/usr/bin/env python3
"""REQ-REPORT-8233: direct CLI imports this checkout from private working directories."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v711_capstone import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
