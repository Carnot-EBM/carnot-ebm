"""REQ-REPORT-8660: direct entry for owned installed compatibility qualification."""

from pathlib import Path
import sys

print("[8660] start no_model_load MODEL_SPECS=[] current_language_model_calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.installed_report_8660 import main

if __name__ == "__main__":
    raise SystemExit(main())
