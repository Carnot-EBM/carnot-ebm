"""REQ-REPORT-8431: a thin numeric-only entry point leaves generator weights untouched."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

from carnot.reporting.v726_criterion_energy_runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
