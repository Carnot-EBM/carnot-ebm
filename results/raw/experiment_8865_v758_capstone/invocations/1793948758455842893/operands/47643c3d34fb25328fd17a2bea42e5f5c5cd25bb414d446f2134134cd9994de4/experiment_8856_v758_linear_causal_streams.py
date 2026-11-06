"""REQ-REPORT-8856: execute private causal streams or authenticate their cold replay."""

from pathlib import Path
import sys

print("[exp8856] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.linear_causal_runner_8856 import main

print("[exp8856] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
