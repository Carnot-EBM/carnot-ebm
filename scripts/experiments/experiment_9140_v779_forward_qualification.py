#!/usr/bin/env python3
"""REQ-REPORT-9140-V779: publish complete-scope forward qualification evidence."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from carnot.reporting.forward_qualification_9140 import main

if __name__ == "__main__":
    raise SystemExit(main())
