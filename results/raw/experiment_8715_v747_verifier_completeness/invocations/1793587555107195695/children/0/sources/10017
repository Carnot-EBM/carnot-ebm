"""REQ-REPORT-8656: resolve this checkout for standalone private execution."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.streaming_execution_runner_8656 import main

if __name__ == "__main__":
    raise SystemExit(main())
