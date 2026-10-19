#!/usr/bin/env python3
"""REQ-REPORT-8083: make the custody CLI runnable from outside the checkout."""

from pathlib import Path
import sys

print("Exp8083 start: V700 contract custody; current model calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v700_custody_execution import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
