"""REQ-REPORT-8582: train only the original small spline head through sealed feedback."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.causal_brier_runner_8582 import main

if __name__ == "__main__":
    raise SystemExit(main())
