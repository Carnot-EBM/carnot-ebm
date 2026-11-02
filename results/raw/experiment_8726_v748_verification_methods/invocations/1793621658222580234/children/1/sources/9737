"""REQ-REPORT-8181: select this checkout even outside its working directory."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.sentence_transport_canary_8181 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
