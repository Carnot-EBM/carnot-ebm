"""REQ-REPORT-8579: direct CLI selects the registered current methods and bounded runner."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v737_contract_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
