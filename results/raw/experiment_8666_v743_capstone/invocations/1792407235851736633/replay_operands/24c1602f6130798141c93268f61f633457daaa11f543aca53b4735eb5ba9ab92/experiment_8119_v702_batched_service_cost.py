"""REQ-REPORT-8119: script execution resolves this checkout outside its cwd."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8119_v702_batched_service_cost import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
