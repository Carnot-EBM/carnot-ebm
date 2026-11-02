#!/usr/bin/env python3
"""REQ-REPORT-7976: expose complete CPU service and private replay commands."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_7976_v691_service_cost import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
