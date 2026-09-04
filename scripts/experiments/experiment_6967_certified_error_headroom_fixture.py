#!/usr/bin/env python3
"""Build the exact V609 error fixture and unused-pair slices.

Spec ref: REQ-VERIFY-6967.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6967_certified_error_headroom_fixture import main


if __name__ == "__main__":  # pragma: no cover - this file is the required command surface.
    raise SystemExit(main())
