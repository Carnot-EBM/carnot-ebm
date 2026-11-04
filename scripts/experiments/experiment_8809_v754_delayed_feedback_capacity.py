"""REQ-REPORT-8809: run or cold-replay the numerical capacity study without an LLM."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.delayed_feedback_capacity_8809 import main

if __name__ == "__main__":
    raise SystemExit(main())
