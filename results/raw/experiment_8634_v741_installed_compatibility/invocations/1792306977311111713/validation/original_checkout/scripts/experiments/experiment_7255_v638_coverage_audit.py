#!/usr/bin/env python3
"""Run the Exp7255 bounded-coverage audit."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from carnot.experiment_7255_v638_coverage_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
