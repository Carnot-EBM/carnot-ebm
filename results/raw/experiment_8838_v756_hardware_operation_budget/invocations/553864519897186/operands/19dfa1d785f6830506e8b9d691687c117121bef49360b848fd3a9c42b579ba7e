"""REQ-REPORT-8838: standalone audit and cold replay without PYTHONPATH."""

from pathlib import Path
import sys

print("[exp8838] phase=imports_before", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.hardware_operation_budget_8838 import main

print("[exp8838] phase=imports_after", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
