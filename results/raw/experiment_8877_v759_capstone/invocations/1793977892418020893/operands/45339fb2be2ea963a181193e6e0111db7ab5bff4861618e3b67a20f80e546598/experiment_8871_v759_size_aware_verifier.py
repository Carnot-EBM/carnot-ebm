"""REQ-REPORT-8871: run or cold-replay the installed size-aware qualification."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.size_aware_verifier_8871 import main

if __name__ == "__main__":
    raise SystemExit(main())
