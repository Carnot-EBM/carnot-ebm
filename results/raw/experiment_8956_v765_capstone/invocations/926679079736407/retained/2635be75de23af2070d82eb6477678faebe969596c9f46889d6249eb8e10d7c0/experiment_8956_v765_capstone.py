"""REQ-REPORT-8956: permit cold replay from outside the checkout without environment setup."""

from pathlib import Path
import sys

print("[exp8956] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.v765_capstone import main

print("[exp8956] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
