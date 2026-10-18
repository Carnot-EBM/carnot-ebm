#!/usr/bin/env python3
"""Experiment 3392: Archive v311, Activate v312.

Spec coverage: REQ-REPORT-3392
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.archive_v311_activate_v312_3392 import write_artifact

def main() -> int:
    path = write_artifact()
    print(path.read_text())
    return 0

if __name__ == "__main__":
    sys.exit(main())
