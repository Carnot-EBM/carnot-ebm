#!/usr/bin/env python3
"""REQ-REPORT-7989: bootstrap private requests and replay from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_7989_v692_service_cost import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
