#!/usr/bin/env python3
"""REQ-REPORT-7994: resolve imports for private external-CWD cold replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_7994_v693_development_cohort import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
