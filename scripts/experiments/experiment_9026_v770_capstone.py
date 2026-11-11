"""REQ-REPORT-9026-SCOPED: run authenticated accounting from any working directory."""

from pathlib import Path
import sys

print("[exp9026] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.v770_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
