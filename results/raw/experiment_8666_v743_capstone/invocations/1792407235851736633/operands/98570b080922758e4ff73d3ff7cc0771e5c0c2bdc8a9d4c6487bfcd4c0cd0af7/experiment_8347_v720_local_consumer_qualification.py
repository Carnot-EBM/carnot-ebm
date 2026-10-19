"""REQ-REPORT-8347: direct entry keeps experiment imports independent of cwd."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.local_consumer_execution_8347 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
