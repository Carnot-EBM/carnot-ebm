"""REQ-VERIFY-8705: direct execution pins imports to this checkout before qualification."""

from pathlib import Path
import sys

print("[exp8705] phase=start no_model_load MODEL_SPECS=[] language_model_calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.arc_decision_qualification_runner_8705 import main

if __name__ == "__main__":
    raise SystemExit(main())
