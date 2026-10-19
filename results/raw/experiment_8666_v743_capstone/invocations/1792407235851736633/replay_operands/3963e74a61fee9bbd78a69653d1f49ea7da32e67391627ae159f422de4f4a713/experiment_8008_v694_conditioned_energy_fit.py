#!/usr/bin/env python3
"""REQ-REPORT-8008: run the small head from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8008_v694_conditioned_energy_fit import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
