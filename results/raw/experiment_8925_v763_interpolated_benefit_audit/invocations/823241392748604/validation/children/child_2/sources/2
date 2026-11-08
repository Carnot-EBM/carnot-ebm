"""REQ-REPORT-8925: run the dated cold audit from any working directory."""

from pathlib import Path
import sys

print("[exp8925] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.interpolated_benefit_runner_8925 import main

print("[exp8925] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
