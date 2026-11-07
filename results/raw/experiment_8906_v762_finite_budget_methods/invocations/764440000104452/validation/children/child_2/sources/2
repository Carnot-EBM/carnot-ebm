"""REQ-REPORT-8906: run directly outside the checkout without PYTHONPATH."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.finite_budget_runner_8906 import main

if __name__ == "__main__":
    raise SystemExit(main())
