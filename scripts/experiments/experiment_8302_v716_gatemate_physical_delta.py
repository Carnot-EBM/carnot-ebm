"""REQ-REPORT-8302: direct unbuffered physical evidence CLI resolves its imports."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.gatemate_delta_execution_8302 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
