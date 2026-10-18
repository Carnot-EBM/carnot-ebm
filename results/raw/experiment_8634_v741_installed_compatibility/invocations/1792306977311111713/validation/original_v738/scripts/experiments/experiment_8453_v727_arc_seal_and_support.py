"""REQ-REPORT-8453: bounded custody recovery makes zero current model calls."""

import os
from pathlib import Path
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
_root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_root / "python"), str(_root)]

from carnot.reporting.arc_seal_and_support_runner_8453 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
