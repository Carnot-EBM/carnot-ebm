"""REQ-REPORT-9030-CUSTODY: direct CLI works outside the checkout without PYTHONPATH."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.retained_shard_runner_9030 import main

if __name__ == "__main__":
    raise SystemExit(main())
