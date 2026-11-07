"""REQ-REPORT-8909: run the frozen dated CLI from any working directory."""

from pathlib import Path
import sys

print("[exp8909] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.guarded_causal_runner_8909 import main

print("[exp8909] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
