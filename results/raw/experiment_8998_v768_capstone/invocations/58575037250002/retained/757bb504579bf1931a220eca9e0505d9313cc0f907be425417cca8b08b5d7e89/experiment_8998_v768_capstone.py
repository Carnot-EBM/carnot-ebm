"""REQ-REPORT-8998: run and replay retained accounting outside the checkout."""

from pathlib import Path
import sys

print("[exp8998] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.v768_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
