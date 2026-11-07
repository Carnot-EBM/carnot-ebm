"""REQ-REPORT-8888: execute and cold-replay the complete CPU adaptation study."""

from pathlib import Path
import sys

print("[exp8888] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.matched_adaptation_runner_8888 import main

print("[exp8888] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
