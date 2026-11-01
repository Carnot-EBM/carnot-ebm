#!/usr/bin/env python3
"""REQ-REPORT-8006: expose the independent reader from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.independent_replay_8006 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
