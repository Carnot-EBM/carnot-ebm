"""REQ-REPORT-8654: expose the same bounded runner from any working directory."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.learning_core_runner_8654 import main

if __name__ == "__main__":
    raise SystemExit(main())
