#!/usr/bin/env python3
"""REQ-REPORT-8178: locate qualified custody modules from any working directory.

The existing runner freezes validation argv and publishes checked bytes. This
entry point only resolves repository modules and declares zero model execution.
"""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
print("Exp8178 start: V707 immutable contract custody; current model calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import v707_contract_custody as experiment  # noqa: E402
from carnot.reporting.v700_custody_execution import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(experiment=experiment))
