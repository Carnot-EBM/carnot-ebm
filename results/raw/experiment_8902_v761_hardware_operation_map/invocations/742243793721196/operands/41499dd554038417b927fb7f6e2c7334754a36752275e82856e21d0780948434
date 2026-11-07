"""REQ-REPORT-8858: run or replay private installed exact-verifier qualification."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting.installed_indexed_runner_8858 import main

if __name__ == "__main__":
    raise SystemExit(main())
