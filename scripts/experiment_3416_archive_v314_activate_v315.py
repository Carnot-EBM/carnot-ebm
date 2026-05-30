#!/usr/bin/env python3
"""Experiment 3416: Archive v314, Activate v315.

Spec coverage: REQ-REPORT-3416
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.archive_v314_activate_v315_3416 import write_artifact


def main() -> int:
    path = write_artifact()
    print(path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
