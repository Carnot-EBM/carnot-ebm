"""REQ-REPORT-8617: allow private replay from outside the checkout."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.triviaplus_release_runner_8617 import main

if __name__ == "__main__":
    raise SystemExit(main())
