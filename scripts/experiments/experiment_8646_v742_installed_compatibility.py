"""REQ-REPORT-8646: run separate historical and current installed qualification."""

from pathlib import Path
import sys

print("[8646] start no_model_load MODEL_SPECS=[] current_language_model_calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.installed_report_8646 import main

if __name__ == "__main__":
    raise SystemExit(main())
