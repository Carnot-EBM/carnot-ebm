"""REQ-REPORT-8244: direct CLI resolves this checkout from any caller directory."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.kv260_decision_execution_8244 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
