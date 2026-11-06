"""REQ-REPORT-8862: run durable linear costs and replay outside the checkout."""

from pathlib import Path
import sys

print("[exp8862] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.linear_durable_runner_8862 import main

print("[exp8862] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
