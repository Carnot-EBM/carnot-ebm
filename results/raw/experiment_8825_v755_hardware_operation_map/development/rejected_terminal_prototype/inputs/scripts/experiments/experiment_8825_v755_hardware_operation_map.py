"""REQ-REPORT-8825: standalone source audit and outside-checkout replay."""

from pathlib import Path
import sys

print("[exp8825] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.hardware_operation_map_8825 import main

print("[exp8825] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
