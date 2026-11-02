#!/usr/bin/env python3
"""REQ-REPORT-8065: run the independent audit from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8065_v698_fresh_learning_audit import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
