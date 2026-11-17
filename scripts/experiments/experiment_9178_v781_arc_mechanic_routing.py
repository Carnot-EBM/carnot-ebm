"""REQ-REPORT-9178-V781: standalone CPU routing controls, with no model invocation."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.arc_mechanic_routing_9178 import main

if __name__ == "__main__":
    raise SystemExit(main())
