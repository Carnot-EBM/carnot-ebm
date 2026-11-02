#!/usr/bin/env python3
"""Experiment 3447: Archive milestones .316 and .317, activate .318.

Spec coverage: REQ-REPORT-3447
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.archive_v316_v317_activate_v318_3447 import write_artifact


def main() -> int:
    path = write_artifact()
    print(path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
