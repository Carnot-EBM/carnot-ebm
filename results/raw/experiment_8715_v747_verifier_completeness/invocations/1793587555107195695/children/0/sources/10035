"""REQ-REPORT-8694: standalone imports bind the capstone to its owned checkout."""

from pathlib import Path
import sys

print("[exp8694] phase=start no_model_load current_language_model_calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.v745_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
