#!/usr/bin/env python3
"""Run Exp 2993 AquaForte/BEAVER substrate corrigendum."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.aquaforte_beaver_substrate_corrigendum import main


if __name__ == "__main__":
    raise SystemExit(main())
