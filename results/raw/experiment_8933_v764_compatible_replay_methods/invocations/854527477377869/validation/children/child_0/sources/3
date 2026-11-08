"""REQ-REPORT-8933: run the CPU replay methods CLI from any working directory."""

import os
from pathlib import Path
import sys

print("[exp8933] phase=imports_before completed=0 pending=1", flush=True)
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.compatible_replay_runner_8933 import main

print("[exp8933] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
