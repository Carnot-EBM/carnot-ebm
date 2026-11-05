"""REQ-REPORT-8835: direct scheduled producer and outside-checkout replay."""

from pathlib import Path
import sys

print("[exp8835] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.complete_adaptation_runner_8835 import main

print("[exp8835] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
