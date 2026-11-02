"""REQ-REPORT-8724: execute the original core with current invocation clocks."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.invocation_clock_core_8724 import main

if __name__ == "__main__":
    raise SystemExit(main())
