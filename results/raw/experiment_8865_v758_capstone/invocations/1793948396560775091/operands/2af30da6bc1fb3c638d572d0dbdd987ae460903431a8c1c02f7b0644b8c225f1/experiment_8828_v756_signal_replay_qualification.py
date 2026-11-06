"""REQ-REPORT-8828: run bounded original replay from any working directory."""

from pathlib import Path
import sys

print("[exp8828] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.signal_replay_runner_8828 import main

print("[exp8828] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
