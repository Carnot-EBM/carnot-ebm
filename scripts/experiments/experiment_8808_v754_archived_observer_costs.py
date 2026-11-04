"""REQ-VERIFY-8808: script-path execution needs no ambient PYTHONPATH."""

from pathlib import Path
import os
import sys

print("[exp8808] phase=imports_before no_model_load MODEL_SPECS=[]", flush=True)
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.archived_observer_runner_8808 import main

print("[exp8808] phase=imports_after language_model_calls=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
