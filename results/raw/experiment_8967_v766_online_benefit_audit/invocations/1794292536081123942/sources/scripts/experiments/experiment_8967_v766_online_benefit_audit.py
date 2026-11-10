"""REQ-REPORT-8967: execute the frozen real-response benefit audit."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.online_benefit_runner_8967 import main

if __name__ == "__main__":
    raise SystemExit(main())
