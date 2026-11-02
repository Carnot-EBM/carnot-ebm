"""REQ-REPORT-8715: audit constructed verifier faults without loading a model."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.verifier_completeness_runner_8715 import main

if __name__ == "__main__":
    raise SystemExit(main())
