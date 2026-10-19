#!/usr/bin/env python3
"""REQ-REPORT-7997: expose frozen CPU decisions without ambient PYTHONPATH."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_7997_v693_typed_development_decisions import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
