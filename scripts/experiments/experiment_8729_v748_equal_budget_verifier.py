"""REQ-REPORT-8729: publish the equal-call constructed audit with zero model loads."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.equal_budget_verifier_runner_8729 import main

if __name__ == "__main__":
    raise SystemExit(main())
