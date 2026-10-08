"""REQ-REPORT-8258: resolve this checkout for the real standalone runner."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.kv260_evidence_cost_execution_8258 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
