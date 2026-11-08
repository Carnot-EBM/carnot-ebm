"""REQ-REPORT-8912: invoke the frozen benchmark outside checkout without PYTHONPATH."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.complete_request_cost_8912 import main

if __name__ == "__main__":
    raise SystemExit(main())
