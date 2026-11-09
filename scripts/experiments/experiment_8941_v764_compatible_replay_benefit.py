"""REQ-REPORT-8941: run the dated read-only audit from any working directory."""

from pathlib import Path
import sys

print("[exp8941] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.compatible_replay_benefit_runner_8941 import main

print("[exp8941] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
