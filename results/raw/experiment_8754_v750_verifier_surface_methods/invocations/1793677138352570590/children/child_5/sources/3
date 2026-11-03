"""REQ-REPORT-8754: qualify fixed verifier methods without loading a model."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.verifier_surface_runner_8754 import main

if __name__ == "__main__":
    raise SystemExit(main())
