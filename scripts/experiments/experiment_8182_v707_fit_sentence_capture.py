"""REQ-REPORT-8182: run this checkout directly from any working directory."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.fit_sentence_capture_8182 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
