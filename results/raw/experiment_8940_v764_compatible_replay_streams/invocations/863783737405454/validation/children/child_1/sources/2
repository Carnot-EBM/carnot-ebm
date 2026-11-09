"""REQ-REPORT-8940: run the CPU replay stream CLI from any working directory."""

import os
from pathlib import Path
import sys

print("[exp8940] phase=imports_before completed=0 pending=1", flush=True)
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.compatible_replay_stream_runner_8940 import main

print("[exp8940] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
