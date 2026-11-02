"""REQ-REPORT-8726: expose the frozen CPU protocol without loading a model."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.verification_methods_runner_8726 import main

if __name__ == "__main__":
    raise SystemExit(main())
