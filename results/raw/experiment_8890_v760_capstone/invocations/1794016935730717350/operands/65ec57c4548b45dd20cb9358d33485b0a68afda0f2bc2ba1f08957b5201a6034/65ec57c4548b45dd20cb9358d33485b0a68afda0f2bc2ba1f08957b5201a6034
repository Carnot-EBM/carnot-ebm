"""REQ-REPORT-8880: direct and foreign-directory invocation share the checked producer."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.sequential_certificate_runner_8880 import main

if __name__ == "__main__":
    raise SystemExit(main())
