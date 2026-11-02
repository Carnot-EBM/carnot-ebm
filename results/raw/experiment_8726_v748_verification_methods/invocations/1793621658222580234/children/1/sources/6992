#!/usr/bin/env python3
"""Run Exp 2257 ODAR real Tier 0 probe benchmark.

Spec: REQ-ODAR-2257, SCENARIO-ODAR-2257
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.reporting.odar_real_benchmark import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
