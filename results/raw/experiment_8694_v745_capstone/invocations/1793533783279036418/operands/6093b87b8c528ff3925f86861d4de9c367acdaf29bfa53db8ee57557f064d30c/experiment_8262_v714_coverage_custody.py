#!/usr/bin/env python3
"""REQ-REPORT-8262: direct imports let private cold readers run after cleanup."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v714_coverage_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
