"""REQ-REPORT-8816: seal prospective selective evidence without opening study predictions."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.selective_runner_8816 import main

if __name__ == "__main__":
    raise SystemExit(main())
