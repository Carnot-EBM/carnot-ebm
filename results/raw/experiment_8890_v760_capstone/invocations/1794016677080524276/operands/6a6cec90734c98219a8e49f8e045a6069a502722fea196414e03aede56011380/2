"""REQ-REPORT-8886: run the dated read-only audit or reconstruct saved evidence."""

from pathlib import Path
import sys

print("[exp8886] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.label_benefit_runner_8886 import main

print("[exp8886] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
