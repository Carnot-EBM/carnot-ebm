"""REQ-VERIFY-8783: run durable identity qualification without model loading."""

from pathlib import Path
import os
import sys

print("[exp8783] phase=start no_model_load MODEL_SPECS=[] language_model_calls=0", flush=True)
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.durable_producer_runner_8783 import main

if __name__ == "__main__":
    raise SystemExit(main())
