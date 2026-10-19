"""REQ-REPORT-8640: keep the direct entrypoint executable outside the checkout."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v742_contract_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
