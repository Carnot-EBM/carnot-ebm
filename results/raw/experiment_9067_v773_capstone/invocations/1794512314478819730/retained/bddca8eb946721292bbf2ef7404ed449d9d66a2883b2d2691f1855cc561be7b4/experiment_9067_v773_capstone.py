"""REQ-REPORT-9067-V773: run or replay exact accounting from any working directory."""

from pathlib import Path
import sys

print("[exp9067] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.v773_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
