"""REQ-REPORT-8540: keep the direct CLI thin so readers can reuse the same methods."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v734_contract_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
