"""REQ-REPORT-8571: direct research CLI keeps worker stdout available for frames."""

from pathlib import Path
import sys

print(
    "[exp8571] phase=start no_model_load MODEL_SPECS=[] current_LLM_calls=0",
    file=sys.stderr,
    flush=True,
)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.resident_runner_8571 import main

if __name__ == "__main__":
    raise SystemExit(main())
