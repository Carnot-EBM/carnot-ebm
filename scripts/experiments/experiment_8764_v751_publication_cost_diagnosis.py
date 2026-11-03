"""REQ-REPORT-8764: allow fresh replay from an external working directory."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.publication_cost_8764 import main

if __name__ == "__main__":
    raise SystemExit(main())
