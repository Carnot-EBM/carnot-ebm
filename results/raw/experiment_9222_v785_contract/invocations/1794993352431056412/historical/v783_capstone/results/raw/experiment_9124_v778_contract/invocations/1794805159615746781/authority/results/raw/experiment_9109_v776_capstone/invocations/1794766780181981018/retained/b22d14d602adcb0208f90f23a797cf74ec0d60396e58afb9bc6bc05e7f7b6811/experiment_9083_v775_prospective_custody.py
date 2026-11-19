"""REQ-REPORT-9083-V775: qualify prospective custody without loading an LLM."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.prospective_custody_runner_9083 import main

if __name__ == "__main__":
    raise SystemExit(main())
