"""REQ-REPORT-8485: thin entry keeps qualification in reusable reporting modules."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v731_contract_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
