"""REQ-REPORT-8436: primitive attribution uses no current language model calls."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
_root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_root / "python"), str(_root)]

from carnot.reporting.transaction_attribution_runner_8436 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
