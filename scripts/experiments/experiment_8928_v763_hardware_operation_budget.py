"""REQ-REPORT-8928: run and cold-replay the actual producer outside the checkout."""

from pathlib import Path
import sys

print("[exp8928] phase=imports_before", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.hardware_operation_budget_8928 import main

print("[exp8928] phase=imports_after", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
