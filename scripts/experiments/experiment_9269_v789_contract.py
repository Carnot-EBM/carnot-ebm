"""REQ-REPORT-9269-V789: qualify split V782-V787 consumer replay, coverage-list parity and V789 contract without loading a model."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v789_contract import main

MODEL_SPECS: list[dict[str, str]] = []

if __name__ == "__main__":
    raise SystemExit(main())
