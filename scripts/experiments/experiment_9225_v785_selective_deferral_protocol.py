"""REQ-REPORT-9225-V785: qualify selective-deferral feedback masks and spline-support diagnostics."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.selective_deferral_runner_9225 import main

if __name__ == "__main__":
    raise SystemExit(main())
