#!/usr/bin/env python3
"""REQ-REPORT-8085: permit private and cold CLI use outside the checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8085_v700_radial_memory_kernel import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
