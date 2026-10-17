"""REQ-REPORT-8622: direct execution works outside the repository working directory."""

from pathlib import Path
import sys

print("[8622] start no_model_load MODEL_SPECS=[] current_language_model_calls=0", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.lean_cost_runner_8622 import main

if __name__ == "__main__":
    raise SystemExit(main())
