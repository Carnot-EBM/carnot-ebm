"""REQ-REPORT-8079: run the owned lifecycle partition outside the checkout."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8079_v699_feature_cache_lifecycle import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
