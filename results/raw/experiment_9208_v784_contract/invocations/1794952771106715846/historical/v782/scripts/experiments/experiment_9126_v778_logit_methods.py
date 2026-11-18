"""REQ-REPORT-9126-V778: execute CPU methods controls with zero model access."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.logit_methods_runner_9126 import main

if __name__ == "__main__":
    raise SystemExit(main())
