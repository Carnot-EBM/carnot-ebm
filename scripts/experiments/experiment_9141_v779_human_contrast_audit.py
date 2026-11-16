"""REQ-REPORT-9141-V779: measure human disagreement without loading a model."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.human_contrast_runner_9141 import main

if __name__ == "__main__":
    raise SystemExit(main())
