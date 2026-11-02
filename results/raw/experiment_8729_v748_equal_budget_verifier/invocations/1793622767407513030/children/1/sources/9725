#!/usr/bin/env python3
"""REQ-REPORT-8164: resolve custody code without depending on the working directory.

The existing runner owns validation and atomic publication. This entry point
only locates those qualified modules and declares that no model will load.
"""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
print("Exp8164 start: V706 contract custody; current model calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import v706_contract_custody as experiment  # noqa: E402
from carnot.reporting.v700_custody_execution import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(experiment=experiment))
