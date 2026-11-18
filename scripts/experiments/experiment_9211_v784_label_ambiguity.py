"""REQ-REPORT-9211-V784: qualify human target identifiability and exact label-ambiguity bounds."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.label_ambiguity_runner_9211 import main

if __name__ == "__main__":
    raise SystemExit(main())
