"""REQ-VERIFY-8745: launch bounded durability admission with zero model calls."""

from pathlib import Path
import os
import sys

print("[exp8745] phase=start no_model_load MODEL_SPECS=[] language_model_calls=0", flush=True)
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.bounded_durable_runner_8745 import main

if __name__ == "__main__":
    raise SystemExit(main())
