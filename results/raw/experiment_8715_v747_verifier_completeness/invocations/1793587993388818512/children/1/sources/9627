"""REQ-REPORT-8055: direct script execution preserves CPU and import settings."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8055_v697_hardware_guard_boundary import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
