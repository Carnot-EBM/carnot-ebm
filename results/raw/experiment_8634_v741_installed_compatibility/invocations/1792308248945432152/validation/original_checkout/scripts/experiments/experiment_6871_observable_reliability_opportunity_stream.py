#!/usr/bin/env python3
"""Run the primary-receipt observable reliability stream build.

Spec ref: REQ-LEARN-6871.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6871_observable_reliability_opportunity_stream import main


if __name__ == "__main__":
    raise SystemExit(main())
