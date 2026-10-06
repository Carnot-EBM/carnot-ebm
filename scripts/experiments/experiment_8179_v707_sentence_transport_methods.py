"""REQ-REPORT-8179: run this exact checkout from any working directory.

Explicit import paths keep an installed Carnot copy from supplying different
protocol or publication code when a cold replay has no PYTHONPATH.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.sentence_transport_methods_8179 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
