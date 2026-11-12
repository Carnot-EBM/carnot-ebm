"""REQ-REPORT-9042-V772: run the no-model prospective graph custody freeze."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.fresh_seal_runner_9042 import main

if __name__ == "__main__":
    raise SystemExit(main())
