#!/usr/bin/env python3
"""REQ-REPORT-8111: a thin script resolves package imports outside the checkout."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
print("Exp8111 start: methods and stream custody; current model calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.methods_stream_execution_8111 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
