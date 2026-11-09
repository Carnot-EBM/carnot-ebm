"""REQ-REPORT-8943: allow authenticated replay from outside the checkout."""

from pathlib import Path
import sys

print("[exp8943] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.v764_capstone import main

print("[exp8943] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
