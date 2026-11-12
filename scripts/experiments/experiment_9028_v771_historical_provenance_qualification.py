"""REQ-REPORT-9028-QUALIFICATION: run preserved-input custody from any directory."""

from pathlib import Path
import sys

print("[exp9028] phase=imports_before completed=0 pending=1", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.historical_provenance_9028 import main

if __name__ == "__main__":
    raise SystemExit(main())
