"""REQ-REPORT-8153: run this checkout from any working directory.

An explicit import root prevents an unrelated installed Carnot from producing
this experiment's evidence when the script is invoked outside the checkout.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.fit_evidence_capture_8153 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
