"""REQ-REPORT-8931: run frozen methods from any working directory."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.graph_source_methods_8931 import main

if __name__ == "__main__":
    raise SystemExit(main())
