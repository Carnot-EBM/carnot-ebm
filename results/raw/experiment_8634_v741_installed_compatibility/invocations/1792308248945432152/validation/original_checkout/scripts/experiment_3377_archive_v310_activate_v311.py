#!/usr/bin/env python3
"""Experiment 3377: Archive v310, Activate v311.

Spec coverage: REQ-REPORT-3377
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.archive_v310_activate_v311_3377 import write_artifact

def main() -> int:
    path = write_artifact()
    print(path.read_text())
    return 0

if __name__ == "__main__":
    sys.exit(main())
