"""REQ-REPORT-8934: expose the same runner from any working directory."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.depth_control_runner_8934 import main

if __name__ == "__main__":
    raise SystemExit(main())
