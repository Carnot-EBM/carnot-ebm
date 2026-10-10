"""REQ-REPORT-8373: bind the standalone capstone to the repository's qualified readers."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v721_capstone import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
