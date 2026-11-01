#!/usr/bin/env python3
"""REQ-REPORT-8084: resolve the real checkout from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8084_v700_fresh_cohort_methods import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
