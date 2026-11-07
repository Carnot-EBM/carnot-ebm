"""REQ-REPORT-8864: cold execution must work outside the source checkout."""

from pathlib import Path
import sys

print("[exp8864] phase=imports_before", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.hardware_budget_8864 import main

print("[exp8864] phase=imports_after", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
