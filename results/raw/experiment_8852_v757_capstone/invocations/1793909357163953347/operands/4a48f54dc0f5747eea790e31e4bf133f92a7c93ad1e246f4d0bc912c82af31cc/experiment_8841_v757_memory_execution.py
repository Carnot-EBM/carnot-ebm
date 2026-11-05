"""REQ-REPORT-8841: private execution qualification and outside-checkout replay."""

from pathlib import Path
import sys

print("[exp8841] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.memory_execution_8841 import main

print("[exp8841] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
