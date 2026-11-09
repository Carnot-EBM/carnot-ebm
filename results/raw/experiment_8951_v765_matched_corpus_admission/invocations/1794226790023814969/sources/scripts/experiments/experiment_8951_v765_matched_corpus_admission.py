"""REQ-REPORT-8951: run the admission producer from any working directory."""

from pathlib import Path
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.matched_admission_runner_8951 import main

if __name__ == "__main__":
    raise SystemExit(main())
