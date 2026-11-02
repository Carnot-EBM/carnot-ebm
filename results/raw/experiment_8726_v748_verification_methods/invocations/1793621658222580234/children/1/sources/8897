#!/usr/bin/env python3
"""Start the independent certified-selection audit in a clean process.

Spec ref: REQ-VERIFY-6960.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6960_certified_selection_cold_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
