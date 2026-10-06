"""REQ-REPORT-8194: run the qualified CPU module from any caller directory.

Explicit checkout paths remove reliance on an ambient PYTHONPATH. The module
trains small heads against cached candidates and loads no language model.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.selective_methods_8194 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
