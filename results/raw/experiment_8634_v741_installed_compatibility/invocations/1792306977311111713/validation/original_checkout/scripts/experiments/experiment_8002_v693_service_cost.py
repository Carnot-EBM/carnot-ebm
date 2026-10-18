#!/usr/bin/env python3
"""REQ-REPORT-8002: expose private CLI paths without an ambient PYTHONPATH."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8002_v693_service_cost import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
