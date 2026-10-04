#!/usr/bin/env python3
"""REQ-REPORT-8105: expose the numerical CLI from outside the checkout."""

import os
from pathlib import Path
import sys

print("[exp8105] start: native radial fixture qualification; model calls=0", flush=True)
os.environ["PYTHONUNBUFFERED"] = "1"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
if os.environ.get("CARNOT_8105_COVERAGE_START"):
    import coverage

    os.environ["COVERAGE_PROCESS_START"] = os.environ["CARNOT_8105_COVERAGE_START"]
    os.environ["COVERAGE_FILE"] = str(
        Path(os.environ["CARNOT_8105_COVERAGE_START"]).parent / ".coverage"
    )
    if coverage.Coverage.current() is None:
        coverage.process_startup()
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.experiment_8105_v701_native_radial_kernel import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
