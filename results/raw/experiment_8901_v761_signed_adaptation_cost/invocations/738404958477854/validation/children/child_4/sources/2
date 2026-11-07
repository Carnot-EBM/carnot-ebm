"""REQ-REPORT-8901: measure and replay complete signed-feature adaptation costs."""

from pathlib import Path
import signal
import sys

print("[exp8901] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.signed_adaptation_runner_8901 import main

print("[exp8901] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    signal.alarm(4800)
    raise SystemExit(main())
