"""REQ-REPORT-8155: bind this checkout when invoked outside its working directory.

Explicit roots prevent an installed package from silently producing evidence
with different code than the frozen local capture adapter.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ["PYTHONUNBUFFERED"] = "1"
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.reserved_evidence_capture_8155 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
