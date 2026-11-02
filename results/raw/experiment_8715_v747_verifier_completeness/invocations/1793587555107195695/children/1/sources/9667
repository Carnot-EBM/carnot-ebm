#!/usr/bin/env python3
"""REQ-REPORT-8116: resolve the qualified CLI independently of working directory."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
if os.environ.get("COVERAGE_PROCESS_START"):
    import coverage

    coverage.process_startup()

from carnot.reporting.independent_memory_execution_8116 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
