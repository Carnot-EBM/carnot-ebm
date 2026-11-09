"""REQ-REPORT-8952: execute the frozen numerical evaluator from any directory."""

from pathlib import Path
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.matched_decisions_runner_8952 import main

if __name__ == "__main__":
    raise SystemExit(main())
