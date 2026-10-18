#!/usr/bin/env python3
"""Run the exact prefix-viability relation fixture.

Spec ref: REQ-CONSTRAINT-6919.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6919_exact_prefix_viability_fixture import main


if __name__ == "__main__":
    raise SystemExit(main())
