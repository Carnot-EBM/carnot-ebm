"""REQ-REPORT-8159: make direct script execution resolve this checkout's code."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.durable_batch_execution_8159 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
