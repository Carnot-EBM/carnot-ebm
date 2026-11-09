"""REQ-REPORT-8955: run complete replay costs from any current directory."""

from pathlib import Path
import sys

print("[exp8955] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.gradient_complete_cost_runner_8955 import main

print("[exp8955] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
