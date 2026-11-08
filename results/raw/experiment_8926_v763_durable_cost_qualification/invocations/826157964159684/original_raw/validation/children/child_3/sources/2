"""REQ-REPORT-8914: run durable service or cold replay from any directory."""

from pathlib import Path
import sys

print("[exp8914] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.guarded_durable_runner_8914 import main

print("[exp8914] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
