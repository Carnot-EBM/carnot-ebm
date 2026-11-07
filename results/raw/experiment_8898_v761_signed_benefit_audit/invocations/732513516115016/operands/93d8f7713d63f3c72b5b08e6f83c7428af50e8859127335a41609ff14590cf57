"""REQ-REPORT-8884: run the dated protocol or reconstruct it outside checkout."""

from pathlib import Path
import sys

print("[exp8884] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.topology_causal_runner_8884 import main

print("[exp8884] phase=imports_after completed=1 pending=0", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
