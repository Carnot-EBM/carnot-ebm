#!/usr/bin/env python3
"""Experiment 3425: Archive v315, Activate v316.

Spec coverage: REQ-REPORT-3425
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.archive_v315_activate_v316_3425 import write_artifact


def main() -> int:
    path = write_artifact()
    print(path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
