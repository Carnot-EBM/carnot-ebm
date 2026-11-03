"""REQ-REPORT-8768: freeze relational evidence-set methods without loading a model."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.evidence_set_runner_8768 import main

if __name__ == "__main__":
    raise SystemExit(main())
