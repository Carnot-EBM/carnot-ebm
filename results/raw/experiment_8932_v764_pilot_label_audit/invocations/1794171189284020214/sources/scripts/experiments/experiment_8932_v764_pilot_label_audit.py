"""REQ-REPORT-8932: make the same audited CLI available outside the checkout."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.pilot_label_audit_runner_8932 import main

if __name__ == "__main__":
    raise SystemExit(main())
