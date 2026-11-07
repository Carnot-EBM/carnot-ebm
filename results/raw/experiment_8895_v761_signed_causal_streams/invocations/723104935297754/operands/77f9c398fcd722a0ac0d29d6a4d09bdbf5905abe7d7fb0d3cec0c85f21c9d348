"""REQ-REPORT-8846: run or cold-replay a read-only future benefit audit."""

from pathlib import Path
import sys

print("[exp8846] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.future_benefit_runner_8846 import main

print("[exp8846] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
