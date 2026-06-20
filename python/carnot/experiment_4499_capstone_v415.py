"""Run Exp 4499 capstone v415 aggregation.

Spec refs: REQ-CAPSTONE-4499, SCENARIO-CAPSTONE-4499.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.capstone_v415_4499 import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
