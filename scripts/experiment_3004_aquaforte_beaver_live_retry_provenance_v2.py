#!/usr/bin/env python3
"""Run Exp 3004 AquaForte/BEAVER live retry provenance repair."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.aquaforte_beaver_live_retry_provenance import main


if __name__ == "__main__":
    raise SystemExit(main())
