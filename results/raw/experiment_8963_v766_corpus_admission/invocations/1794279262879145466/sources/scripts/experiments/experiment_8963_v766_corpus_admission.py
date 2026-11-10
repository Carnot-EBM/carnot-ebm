"""REQ-REPORT-8963: expose admission and replay from outside the checkout."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.corpus_admission_runner_8963 import main

if __name__ == "__main__":
    raise SystemExit(main())
