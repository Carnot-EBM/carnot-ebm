"""REQ-REPORT-8435: bounded conditional audit with no current language model calls."""

import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
_root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_root / "python"), str(_root)]

from carnot.reporting.capacity_benefit_runner_8435 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
