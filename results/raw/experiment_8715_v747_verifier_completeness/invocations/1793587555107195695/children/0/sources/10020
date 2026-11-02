"""REQ-REPORT-8661: direct entry for the fixed complete installed cost matrix."""

from pathlib import Path
import sys

print("[8661] start no_model_load MODEL_SPECS=[] current_language_model_calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.installed_cost_runner_8661 import main

if __name__ == "__main__":
    raise SystemExit(main())
