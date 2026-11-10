"""REQ-REPORT-8984: allow cold replay from any working directory."""

from pathlib import Path
import sys

print("[exp8984] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.v767_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
