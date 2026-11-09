"""REQ-REPORT-8950: expose the independent B collector from any working directory."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.matched_corpus_b_8950 import main

if __name__ == "__main__":
    raise SystemExit(main())
