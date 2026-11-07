"""REQ-REPORT-8881: bootstrap the dated experiment and cold replay outside checkout."""

from pathlib import Path
import sys

print("[exp8881] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.topology_memory_runner_8881 import main

print("[exp8881] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
