"""REQ-REPORT-9244-V787: qualify cluster-valid conformal risk, CA-OCRC tracking and Mycelium provenance."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.conformal_risk_runner_9244 import main

if __name__ == "__main__":
    raise SystemExit(main())
