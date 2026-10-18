"""REQ-REPORT-8432: frozen numeric evaluation performs no current model load."""

import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
_root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_root / "python"), str(_root)]

from carnot.reporting.v726_criterion_decision_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
