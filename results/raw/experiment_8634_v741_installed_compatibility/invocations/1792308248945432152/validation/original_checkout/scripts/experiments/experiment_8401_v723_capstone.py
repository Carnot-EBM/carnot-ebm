"""REQ-REPORT-8401: expose the capstone through a standalone unbuffered runner."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v723_capstone import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
