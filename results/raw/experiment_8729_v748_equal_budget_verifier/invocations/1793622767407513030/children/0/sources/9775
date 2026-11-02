#!/usr/bin/env python3
"""REQ-REPORT-8222: direct CLI exposes authenticated fitting and cold replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.utility_fit_execution_8222 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
