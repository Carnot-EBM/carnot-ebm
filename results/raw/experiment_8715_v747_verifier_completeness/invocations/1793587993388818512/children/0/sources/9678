"""REQ-REPORT-8135: resolve local code when the CLI runs outside the checkout."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v703_capstone import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
