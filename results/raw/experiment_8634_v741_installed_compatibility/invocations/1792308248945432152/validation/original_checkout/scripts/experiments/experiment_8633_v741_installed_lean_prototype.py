"""REQ-REPORT-8633: opt-in installed execution with retained immutable evidence."""

from pathlib import Path
import sys

print("[8633] start no_model_load MODEL_SPECS=[] current_language_model_calls=0", flush=True)

# Direct script execution must resolve publication readers from this checkout.
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.installed_runner_8633 import main

if __name__ == "__main__":
    raise SystemExit(main())
