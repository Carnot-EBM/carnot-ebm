"""REQ-VERIFY-8720: pin imports before running the bounded no-model reader."""

from pathlib import Path
import sys

print("[exp8720] phase=start no_model_load MODEL_SPECS=[] language_model_calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.arc_supervisor_runner_8720 import main

if __name__ == "__main__":
    raise SystemExit(main())
