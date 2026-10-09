"""REQ-REPORT-8357: run the bounded history ledger from any working directory."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("TMPDIR", "/tmp/carnot8357-20261009")
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.gatemate_ledger_execution_8357 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
