#!/usr/bin/env python3
"""REQ-REPORT-8076: make the real CLI independent of its working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8076_v699_projected_online_learning import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
