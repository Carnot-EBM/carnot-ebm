"""REQ-REPORT-8306: restore imports for a direct, unbuffered private CLI."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.local_update_execution_8306 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
