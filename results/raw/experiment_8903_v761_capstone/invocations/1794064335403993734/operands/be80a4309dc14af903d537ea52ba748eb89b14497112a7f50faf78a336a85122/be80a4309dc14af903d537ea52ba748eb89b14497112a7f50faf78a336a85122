"""REQ-REPORT-8896: support direct execution outside the checkout without PYTHONPATH."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.shared_terminal_evaluation_runner_8896 import main

if __name__ == "__main__":
    raise SystemExit(main())
