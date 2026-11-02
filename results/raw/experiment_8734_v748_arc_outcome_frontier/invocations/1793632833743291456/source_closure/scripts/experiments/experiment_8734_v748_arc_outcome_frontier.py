"""REQ-VERIFY-8734: announce startup before importing the model-free reader."""

from pathlib import Path
import sys

print("[exp8734] phase=start no_model_load MODEL_SPECS=[] language_model_calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.arc_outcome_frontier_8734 import main

if __name__ == "__main__":
    raise SystemExit(main())
