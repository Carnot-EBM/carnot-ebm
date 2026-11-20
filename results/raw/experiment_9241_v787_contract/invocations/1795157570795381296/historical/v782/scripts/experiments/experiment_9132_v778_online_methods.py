"""REQ-REPORT-9132-V778: freeze delayed numerical controls with zero model access."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.online_methods_runner_9132 import main

if __name__ == "__main__":
    raise SystemExit(main())
