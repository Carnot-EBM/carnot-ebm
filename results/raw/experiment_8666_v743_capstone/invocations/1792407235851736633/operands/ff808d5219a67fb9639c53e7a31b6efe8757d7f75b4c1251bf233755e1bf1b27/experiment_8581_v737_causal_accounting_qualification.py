"""REQ-REPORT-8581: qualify fresh accounting using the explicit current revision."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.causal_accounting_runner_8581 import main

if __name__ == "__main__":
    raise SystemExit(main())
