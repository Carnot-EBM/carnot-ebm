"""REQ-REPORT-8895: execute frozen signed streams or replay retained causal operands."""

from pathlib import Path
import sys

print("[exp8895] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.signed_causal_runner_8895 import main

print("[exp8895] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
