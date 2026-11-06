"""REQ-REPORT-8865: account for V758 without activation or external publication."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.v758_capstone_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
