#!/usr/bin/env python3
"""REQ-REPORT-8349: enter the measured runner without loading an LLM."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.bounded_feedback_execution_8349 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
