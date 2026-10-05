#!/usr/bin/env python3
"""REQ-REPORT-8151: run immutable source custody without ambient imports."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
print("Exp8151 start: immutable source methods; current model calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.source_method_custody_8151 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
