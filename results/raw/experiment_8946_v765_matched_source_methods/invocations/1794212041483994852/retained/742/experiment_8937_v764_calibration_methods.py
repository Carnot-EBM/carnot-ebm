"""REQ-REPORT-8937: run frozen calibration methods from any working directory."""

import os
from pathlib import Path
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.calibration_methods_8937 import main

if __name__ == "__main__":
    raise SystemExit(main())
