#!/usr/bin/env python3
"""REQ-REPORT-8097: bootstrap the shared custody CLI outside the checkout."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
print("Exp8097 start: V701 contract custody; current model calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import v701_contract_custody as experiment  # noqa: E402
from carnot.reporting.v700_custody_execution import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(experiment=experiment))
