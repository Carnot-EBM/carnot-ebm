"""REQ-REPORT-8620: support direct bounded extraction and cold replay outside checkout."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.triviaplus_extraction_runner_8620 import main

if __name__ == "__main__":
    raise SystemExit(main())
