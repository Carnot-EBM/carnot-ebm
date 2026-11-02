"""REQ-REPORT-8195: run cached CPU fitting from any caller directory.

Explicit paths make private CLI replay independent of ambient PYTHONPATH.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.selective_energy_fit_8195 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
