"""REQ-REPORT-8680: standalone imports bind this runner to its owned checkout."""

from pathlib import Path
import sys

print("[exp8680] phase=start no_model_load current_language_model_calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.v744_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
