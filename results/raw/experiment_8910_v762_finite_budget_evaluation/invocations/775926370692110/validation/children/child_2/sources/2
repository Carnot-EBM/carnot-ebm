"""REQ-REPORT-8910: allow direct cold execution outside the checkout."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.finite_budget_evaluation_runner_8910 import main

if __name__ == "__main__":
    raise SystemExit(main())
