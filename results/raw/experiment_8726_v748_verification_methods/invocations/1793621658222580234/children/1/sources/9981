"""REQ-REPORT-8565: one thin direct CLI keeps consumers on the registered methods."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v736_contract_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
