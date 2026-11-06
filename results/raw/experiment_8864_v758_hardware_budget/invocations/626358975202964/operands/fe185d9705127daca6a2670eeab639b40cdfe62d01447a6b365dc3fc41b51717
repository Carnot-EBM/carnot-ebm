"""REQ-REPORT-8815: direct finite diagnosis and outside-checkout replay."""

from pathlib import Path
import sys

print("[exp8815] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.learning_signal_runner_8815 import main

print("[exp8815] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
