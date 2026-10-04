#!/usr/bin/env python3
"""REQ-REPORT-8137: run source custody without ambient checkout imports."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
print("Exp8137 start: source protocol qualification; current model calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.source_protocol_8137 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
