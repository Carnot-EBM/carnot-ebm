"""REQ-REPORT-9180-V782: freeze fourteen tasks and diagnostic supports without model work."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v782_contract import main

if __name__ == "__main__":
    raise SystemExit(main())
