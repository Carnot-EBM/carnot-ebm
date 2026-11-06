"""REQ-REPORT-8879: run qualification or cold replay with explicit private custody."""

from pathlib import Path
import sys

print("[exp8879] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.diagnosis_qualification_8879 import main

print("[exp8879] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
