"""REQ-VERIFY-8692: standalone CPU execution binds imports to the owned checkout."""

from pathlib import Path
import sys

print("[exp8692] phase=start no_model_load MODEL_SPECS=[] language_model_calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.spline_precision_runner_8692 import main

if __name__ == "__main__":
    raise SystemExit(main())
