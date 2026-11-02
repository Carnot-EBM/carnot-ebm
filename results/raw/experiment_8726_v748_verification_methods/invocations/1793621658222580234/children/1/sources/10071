#!/usr/bin/env python3
"""REQ-REPORT-8716: run bounded installed import isolation without a model load."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.installed_import_runner_8716 import main

if __name__ == "__main__":
    raise SystemExit(main())
