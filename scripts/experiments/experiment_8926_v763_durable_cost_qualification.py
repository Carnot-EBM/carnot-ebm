"""REQ-REPORT-8926: qualify retained costs or cold-replay from any directory."""

from pathlib import Path
import sys

print("[exp8926] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.durable_cost_qualification_8926 import main

print("[exp8926] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
