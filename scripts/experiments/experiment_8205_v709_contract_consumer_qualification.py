#!/usr/bin/env python3
"""REQ-REPORT-8205: run a no-model receipt directly from any working directory."""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
config = os.environ.get("CARNOT_8205_COVERAGE_CONFIG")
if config:
    os.environ["COVERAGE_PROCESS_START"] = config
    import coverage

    coverage.process_startup()
from carnot.reporting.v709_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
