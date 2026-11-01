"""REQ-REPORT-8526: keep the direct CLI thin so readers can reuse the same methods."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v733_contract_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
