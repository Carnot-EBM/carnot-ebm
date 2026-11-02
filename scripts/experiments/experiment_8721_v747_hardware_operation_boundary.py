"""REQ-REPORT-8721: standalone board mapping with zero language-model calls."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.hardware_operation_runner_8721 import main

if __name__ == "__main__":
    raise SystemExit(main())
