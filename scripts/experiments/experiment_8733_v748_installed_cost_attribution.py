#!/usr/bin/env python3
"""REQ-REPORT-8733: expose the version-bound installed cost diagnostic."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.installed_cost_attribution_8733 import main

if __name__ == "__main__":
    raise SystemExit(main())
