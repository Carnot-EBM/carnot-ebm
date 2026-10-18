"""REQ-REPORT-8591: script-path entry retains actual private CLI execution."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v738_contract_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
