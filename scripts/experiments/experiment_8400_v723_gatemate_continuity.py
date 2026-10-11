"""REQ-REPORT-8400: run GateMate continuity with visible, unbuffered boundaries."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.gatemate_continuity_runner_8400 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
