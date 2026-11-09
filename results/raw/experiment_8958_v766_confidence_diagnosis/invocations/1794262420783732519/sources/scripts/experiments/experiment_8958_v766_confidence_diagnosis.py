"""REQ-REPORT-8958: execute the development diagnosis from any directory."""

import os
from pathlib import Path
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.confidence_diagnosis_runner_8958 import main

if __name__ == "__main__":
    raise SystemExit(main())
