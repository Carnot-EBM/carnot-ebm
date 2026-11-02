#!/usr/bin/env python3
"""REQ-REPORT-8624: run the observer qualification from any working directory."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.arc_observer_runner_8624 import main  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []

if __name__ == "__main__":
    raise SystemExit(main())
