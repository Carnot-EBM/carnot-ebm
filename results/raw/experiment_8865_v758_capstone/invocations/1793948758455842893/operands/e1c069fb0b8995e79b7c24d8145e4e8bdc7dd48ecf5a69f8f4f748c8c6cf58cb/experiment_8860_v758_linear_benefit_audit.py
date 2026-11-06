"""REQ-REPORT-8860: run private read-only evidence or an outside-checkout replay."""

from pathlib import Path
import sys

print("[exp8860] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.linear_benefit_runner_8860 import main

print("[exp8860] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
