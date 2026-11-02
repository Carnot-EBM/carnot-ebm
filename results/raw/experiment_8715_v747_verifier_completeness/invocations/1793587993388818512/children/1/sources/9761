#!/usr/bin/env python3
"""REQ-REPORT-8234: resolve this checkout even when replay starts elsewhere."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.decision_margin_runner_8234 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
