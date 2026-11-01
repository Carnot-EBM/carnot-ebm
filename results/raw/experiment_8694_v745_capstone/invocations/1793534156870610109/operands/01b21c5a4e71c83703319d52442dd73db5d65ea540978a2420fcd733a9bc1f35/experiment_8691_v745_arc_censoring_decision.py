"""REQ-VERIFY-8691: standalone execution binds imports to the owned checkout."""

from pathlib import Path
import sys

print("[exp8691] phase=start no_model_load MODEL_SPECS=[] language_model_calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.arc_censoring_runner_8691 import main

if __name__ == "__main__":
    raise SystemExit(main())
