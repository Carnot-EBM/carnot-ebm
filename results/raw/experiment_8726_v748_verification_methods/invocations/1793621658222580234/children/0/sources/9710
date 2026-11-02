"""REQ-REPORT-8149: run from any working directory without PYTHONPATH."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v704_capstone import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
