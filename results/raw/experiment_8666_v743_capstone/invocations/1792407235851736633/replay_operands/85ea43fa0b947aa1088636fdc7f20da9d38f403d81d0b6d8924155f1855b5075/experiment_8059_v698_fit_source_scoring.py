#!/usr/bin/env python3
"""REQ-REPORT-8059: run current capture from an external working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8059_v698_fit_source_scoring import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
