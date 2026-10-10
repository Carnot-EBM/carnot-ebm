"""REQ-REPORT-8362: a direct script path works from outside the checkout."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.threshold_guard_execution_8362 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
