#!/usr/bin/env python3
"""Experiment 3470: Archive milestone .319, activate .320.

Spec coverage: REQ-REPORT-3470
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.archive_v319_activate_v320_3470 import write_artifact


def main() -> int:
    path = write_artifact(repo_root=REPO_ROOT)
    print(path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
