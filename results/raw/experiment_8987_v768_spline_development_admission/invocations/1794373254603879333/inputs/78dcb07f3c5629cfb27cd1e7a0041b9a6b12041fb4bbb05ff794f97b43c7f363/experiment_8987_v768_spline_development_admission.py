"""REQ-REPORT-8987-ADMISSION: run development admission from any working directory."""

import os
from pathlib import Path
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.spline_development_runner_8987 import main

if __name__ == "__main__":
    raise SystemExit(main())
