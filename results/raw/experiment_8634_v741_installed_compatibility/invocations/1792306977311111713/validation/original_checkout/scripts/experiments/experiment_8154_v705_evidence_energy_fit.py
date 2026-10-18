"""REQ-REPORT-8154: run the owned checkout from any working directory.

Explicit paths prevent an installed package from substituting different code
when an operator invokes this script without PYTHONPATH.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.evidence_energy_fit_8154 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
