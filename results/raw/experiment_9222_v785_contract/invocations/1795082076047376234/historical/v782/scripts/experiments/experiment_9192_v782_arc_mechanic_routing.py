"""REQ-REPORT-9192-V782: standalone CPU compatibility study; no model or game load."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.arc_mechanic_routing_9192 import main

if __name__ == "__main__":
    raise SystemExit(main())
