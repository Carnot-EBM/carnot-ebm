"""REQ-VERIFY-8760: announce the model-free phase before loading the reader."""

from pathlib import Path
import sys

print("[exp8760] phase=start no_model_load MODEL_SPECS=[] language_model_calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.arc_horizon_sensitivity_8760 import main

if __name__ == "__main__":
    raise SystemExit(main())
