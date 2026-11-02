#!/usr/bin/env python3
"""REQ-REPORT-8717: measure exact installed batches with no language-model calls."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.installed_batch_cost_runner_8717 import main

if __name__ == "__main__":
    raise SystemExit(main())
