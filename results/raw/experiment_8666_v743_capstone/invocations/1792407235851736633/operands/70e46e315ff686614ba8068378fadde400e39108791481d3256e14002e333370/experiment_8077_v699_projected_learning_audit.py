#!/usr/bin/env python3
"""REQ-REPORT-8077: make private CLI checks independent of working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8077_v699_projected_learning_audit import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
