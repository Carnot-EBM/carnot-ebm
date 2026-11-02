"""REQ-REPORT-8681: run fresh registration from any working directory."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v745_contract_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
