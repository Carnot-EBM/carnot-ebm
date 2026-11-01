"""REQ-REPORT-8566: keep direct invocation bound to the small CPU study adapter."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.count_semantics_runner_8566 import main

if __name__ == "__main__":
    raise SystemExit(main())
