"""REQ-REPORT-8893: direct execution also works outside the checkout without PYTHONPATH."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.shared_terminal_runner_8893 import main

if __name__ == "__main__":
    raise SystemExit(main())
