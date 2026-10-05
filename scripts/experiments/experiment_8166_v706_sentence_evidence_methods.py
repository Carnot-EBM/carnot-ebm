"""REQ-REPORT-8166: resolve this checkout from an outside working directory.

Explicit import roots prevent another installed Carnot copy from changing the
protocol, parser or publication behavior when PYTHONPATH is absent.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.sentence_methods_8166 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
