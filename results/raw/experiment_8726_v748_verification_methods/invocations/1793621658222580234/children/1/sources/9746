#!/usr/bin/env python3
"""REQ-REPORT-8192: resolve qualified custody modules outside the checkout.

The shared runner freezes validation argv and publishes checked bytes. This
entry point declares zero current model calls and leaves live roadmaps untouched.
"""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
print("Exp8192 start: V708 immutable contract custody; current model calls=0", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import v708_contract_custody as experiment  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(
        experiment.run(heartbeat_s=float(os.environ.get("CARNOT_8192_HEARTBEAT_S", "30")))
    )
