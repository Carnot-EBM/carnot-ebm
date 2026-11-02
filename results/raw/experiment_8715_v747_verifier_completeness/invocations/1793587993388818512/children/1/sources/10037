"""REQ-REPORT-8696: execute the sealed original learning core without model loads."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.learning_core_runner_8696 import main

if __name__ == "__main__":
    raise SystemExit(main())
