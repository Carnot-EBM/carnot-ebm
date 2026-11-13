"""REQ-REPORT-9079-V774: expose the same evidence audit from any working directory."""

from pathlib import Path
import sys

print("[exp9079] phase=imports_before", flush=True)
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.hardware_boundary_9079 import main

print("[exp9079] phase=imports_after", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
