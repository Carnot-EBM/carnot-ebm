"""REQ-REPORT-8911: run the frozen dated audit from any working directory."""

from pathlib import Path
import sys

print("[exp8911] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.guarded_benefit_runner_8911 import main

print("[exp8911] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
