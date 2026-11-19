"""REQ-REPORT-9082-V775: publish checked administrative custody without a model load."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v775_contract import main

if __name__ == "__main__":
    raise SystemExit(main())
