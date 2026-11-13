"""REQ-REPORT-9081-V774: expose checked accounting from any working directory."""

from pathlib import Path
import sys

print("[exp9081] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.v774_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
