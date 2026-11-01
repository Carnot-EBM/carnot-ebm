"""REQ-REPORT-8697: qualify original causal controls with zero model calls."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.learning_union_8697 import main

if __name__ == "__main__":
    raise SystemExit(main())
