"""REQ-REPORT-8177: locate repository imports from an external working directory."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v706_capstone import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
