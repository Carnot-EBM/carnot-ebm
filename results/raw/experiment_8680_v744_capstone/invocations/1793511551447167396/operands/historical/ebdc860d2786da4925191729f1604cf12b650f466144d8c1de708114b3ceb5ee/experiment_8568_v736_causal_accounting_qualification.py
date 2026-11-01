"""REQ-REPORT-8568: run fresh causal accounting with the corrected validator."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.causal_accounting_runner_8568 import main

if __name__ == "__main__":
    raise SystemExit(main())
