#!/usr/bin/env python3
"""REQ-REPORT-8402: keep historical custody available from a standalone CLI."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.historical_consumer_runner_8402 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
