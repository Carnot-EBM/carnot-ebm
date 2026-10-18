"""REQ-REPORT-8232: resolve checkout imports for a direct unbuffered audit CLI."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.gatemate_execution_8232 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
