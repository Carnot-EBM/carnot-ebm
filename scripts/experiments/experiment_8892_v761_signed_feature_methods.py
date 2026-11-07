"""REQ-REPORT-8892: run the dated method and cold replay outside the checkout."""

from pathlib import Path
import sys

print("[exp8892] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.signed_feature_runner_8892 import main

print("[exp8892] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
