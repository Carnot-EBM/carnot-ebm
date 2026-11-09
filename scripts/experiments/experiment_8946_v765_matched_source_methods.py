#!/usr/bin/env python3
"""REQ-REPORT-8946: resolve this checkout during foreign-directory cold replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.matched_source_runner_8946 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
