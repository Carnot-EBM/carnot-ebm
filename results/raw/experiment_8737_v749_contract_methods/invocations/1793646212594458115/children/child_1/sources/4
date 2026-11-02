"""REQ-REPORT-8737: expose the bounded no-model V749 methods registration."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v749_contract_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
