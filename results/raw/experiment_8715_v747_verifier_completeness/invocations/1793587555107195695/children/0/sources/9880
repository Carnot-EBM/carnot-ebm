"""REQ-REPORT-8433: numeric synthetic checkpoint qualification loads no model."""

import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
_root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_root / "python"), str(_root)]

from carnot.reporting.block_checkpoint_runner_8433 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
