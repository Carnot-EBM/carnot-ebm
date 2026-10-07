#!/usr/bin/env python3
"""REQ-REPORT-8225: direct runner keeps imports independent of ambient PYTHONPATH."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.delayed_utility_execution_8225 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
