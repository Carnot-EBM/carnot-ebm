#!/usr/bin/env python3
"""Experiment 3391: Archive v312, Activate v313.

Spec coverage: REQ-REPORT-3391
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.archive_v312_activate_v313_3391 import write_artifact


def main() -> int:
    path = write_artifact()
    print(path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
