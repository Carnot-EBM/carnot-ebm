"""REQ-REPORT-8555: expose the dated canonical accounting qualification CLI."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.canonical_update_runner_8555 import main

if __name__ == "__main__":
    raise SystemExit(main())
