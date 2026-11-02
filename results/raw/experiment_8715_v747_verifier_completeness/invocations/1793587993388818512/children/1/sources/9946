"""REQ-REPORT-8545: thin entry point keeps script-path imports independent of cwd."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.nine_input_runner_8545 import main

if __name__ == "__main__":
    raise SystemExit(main())
