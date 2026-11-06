"""REQ-REPORT-8867: run private diagnosis or replay outside the checkout."""

from pathlib import Path
import sys

print("[exp8867] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.feedback_information_runner_8867 import main

print("[exp8867] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
