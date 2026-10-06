"""REQ-REPORT-8183: run the checkout's CPU experiment from any directory.

Explicit paths avoid dependence on a caller's PYTHONPATH or working directory.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.sentence_energy_fit_8183 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
