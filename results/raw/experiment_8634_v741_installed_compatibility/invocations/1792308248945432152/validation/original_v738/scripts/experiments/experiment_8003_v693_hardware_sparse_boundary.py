#!/usr/bin/env python3
"""REQ-REPORT-8003: expose CPU evidence routes without an ambient PYTHONPATH."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8003_v693_hardware_sparse_boundary import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
