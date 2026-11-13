"""REQ-REPORT-9069-V774: authenticate source custody without opening graph truth."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.source_custody_runner_9069 import main

if __name__ == "__main__":
    raise SystemExit(main())
