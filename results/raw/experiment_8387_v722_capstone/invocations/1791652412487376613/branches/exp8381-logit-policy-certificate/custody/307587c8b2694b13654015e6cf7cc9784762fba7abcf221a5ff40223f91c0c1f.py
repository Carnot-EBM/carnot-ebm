"""REQ-REPORT-8381: direct CLI execution works outside the checkout."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.logit_policy_execution_8381 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
