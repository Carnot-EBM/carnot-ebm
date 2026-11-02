"""REQ-REPORT-8559: a direct CLI works without the caller's import environment."""

from pathlib import Path
import sys

print("[exp8559] phase=start no_model_load MODEL_SPECS=[] current_LLM_calls=0", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.complete_native_runner_8559 import main

if __name__ == "__main__":
    raise SystemExit(main())
