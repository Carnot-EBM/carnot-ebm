"""REQ-REPORT-8196: explicit imports let any caller seal cached CPU decisions."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.selective_sealed_evaluation_8196 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
