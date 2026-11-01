#!/usr/bin/env python3
"""REQ-REPORT-8208: real private CLI works without ambient checkout imports."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.restricted_energy_fit_8208 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
