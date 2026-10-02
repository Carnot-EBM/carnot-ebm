#!/usr/bin/env python3
"""REQ-REPORT-8016: run the existing package from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8016_v694_hardware_update_boundary import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
