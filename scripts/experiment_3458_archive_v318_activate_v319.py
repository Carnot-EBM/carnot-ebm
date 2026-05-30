#!/usr/bin/env python3
"""Experiment 3458: Archive milestone .318, activate .319.

Spec coverage: REQ-REPORT-3458
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.archive_v318_activate_v319_3458 import write_artifact


def main() -> int:
    path = write_artifact(repo_root=REPO_ROOT)
    print(path.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
