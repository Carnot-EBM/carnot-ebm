#!/usr/bin/env python3
"""Run the receipt-only V617 capstone from REQ-CAP-7049."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7049_v617_capstone_disposition import main


if __name__ == "__main__":
    raise SystemExit(main())
