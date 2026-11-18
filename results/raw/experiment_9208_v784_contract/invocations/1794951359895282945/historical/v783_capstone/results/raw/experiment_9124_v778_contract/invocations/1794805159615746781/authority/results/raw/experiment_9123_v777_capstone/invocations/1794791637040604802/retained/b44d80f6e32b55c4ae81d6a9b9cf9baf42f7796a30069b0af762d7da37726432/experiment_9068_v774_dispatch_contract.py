"""REQ-REPORT-9068-V774: publish checked dispatch custody without a model load."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v774_dispatch_contract import main

if __name__ == "__main__":
    raise SystemExit(main())
