#!/usr/bin/env python3
"""REQ-REPORT-8027: expose the same numerical measurement from any directory."""

import os
from pathlib import Path
import sys

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.experiment_8027_v695_native_update_cost import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
