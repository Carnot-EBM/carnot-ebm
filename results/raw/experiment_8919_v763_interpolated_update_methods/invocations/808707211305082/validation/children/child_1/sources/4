"""REQ-REPORT-8919: execute the dated CPU methods CLI from any working directory."""

import os
from pathlib import Path
import sys

print("[exp8919] phase=imports_before completed=0 pending=1", flush=True)
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.interpolated_update_runner_8919 import main

print("[exp8919] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
