#!/usr/bin/env python3
"""Run REQ-REPORT-7993 from any directory without ambient PYTHONPATH."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_7993_v693_capture_custody import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
