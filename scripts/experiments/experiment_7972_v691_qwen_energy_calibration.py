#!/usr/bin/env python3
"""REQ-REPORT-7972: direct CPU fitting and cold replay use the checkout library."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_7972_v691_qwen_energy_calibration import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
