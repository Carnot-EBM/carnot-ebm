"""REQ-VERIFY-8796: admit original authority before any observer timing."""

from pathlib import Path
import os
import sys

print("[exp8796] phase=start no_model_load MODEL_SPECS=[] language_model_calls=0", flush=True)
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.observer_intervention_runner_8796 import main

if __name__ == "__main__":
    raise SystemExit(main())
