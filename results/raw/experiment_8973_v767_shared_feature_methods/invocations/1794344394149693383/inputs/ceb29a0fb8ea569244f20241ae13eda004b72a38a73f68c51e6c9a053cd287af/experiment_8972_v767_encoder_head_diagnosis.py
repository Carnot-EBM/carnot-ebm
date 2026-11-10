"""REQ-REPORT-8972: run the dated diagnostic or replay from any working directory."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.encoder_head_runner_8972 import main

if __name__ == "__main__":
    raise SystemExit(main())
