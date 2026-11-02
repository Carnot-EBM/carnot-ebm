"""REQ-REPORT-8176: resolve the checkout for direct execution from any directory."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.hardware_boundary_execution_8176 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
