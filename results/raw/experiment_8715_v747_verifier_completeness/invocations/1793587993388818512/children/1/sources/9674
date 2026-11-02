#!/usr/bin/env python3
"""REQ-REPORT-8124: resolve the no-model worker outside the checkout."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
print("Exp8124 start: seal evidence protocol; current model calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.evidence_protocol_8124 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
