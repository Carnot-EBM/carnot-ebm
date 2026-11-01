"""REQ-REPORT-8316: run the no-probe obligation audit from any working directory."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.gatemate_obligation_execution_8316 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
