"""REQ-REPORT-8854: qualify or cold-replay matched linear causal methods."""

from pathlib import Path
import sys

print("[exp8854] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.linear_ablation_runner_8854 import main

print("[exp8854] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
