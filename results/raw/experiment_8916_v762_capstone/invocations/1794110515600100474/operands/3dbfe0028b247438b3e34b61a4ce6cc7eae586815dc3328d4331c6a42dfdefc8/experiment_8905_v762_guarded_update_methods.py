"""REQ-REPORT-8905: execute the dated methods CLI outside the checkout."""

from pathlib import Path
import sys

print("[exp8905] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.guarded_update_runner_8905 import main

print("[exp8905] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
