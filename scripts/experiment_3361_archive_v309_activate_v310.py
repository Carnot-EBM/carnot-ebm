#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.archive_v309_activate_v310_3361 import write_artifact

def main():
    path = write_artifact()
    print(path.read_text())
    return 0

if __name__ == "__main__":
    sys.exit(main())
