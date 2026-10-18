#!/usr/bin/env python3
"""REQ-REPORT-8123: resolve the custody CLI independently of its working directory."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
print("Exp8123 start: V703 contract custody; current model calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import v703_contract_custody as experiment  # noqa: E402
from carnot.reporting.v700_custody_execution import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(experiment=experiment))
