"""REQ-REPORT-8851: bootstrap cold replay from outside the checkout."""

from pathlib import Path
import sys

print("[exp8851] phase=imports_before", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.hardware_operation_budget_8851 import main

print("[exp8851] phase=imports_after", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
