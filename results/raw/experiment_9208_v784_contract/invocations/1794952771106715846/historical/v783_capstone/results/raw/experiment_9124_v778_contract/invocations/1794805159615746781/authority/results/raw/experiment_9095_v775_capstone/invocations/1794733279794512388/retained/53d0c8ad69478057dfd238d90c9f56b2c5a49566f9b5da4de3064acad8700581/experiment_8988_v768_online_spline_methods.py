"""REQ-REPORT-8988-CAUSAL: run fixed-basis methods from outside the checkout."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.online_spline_runner_8988 import main

if __name__ == "__main__":
    raise SystemExit(main())
