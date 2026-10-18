"""REQ-REPORT-8372: run the read-only reopening ledger with unbuffered progress."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.gatemate_missing_runner_8372 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
