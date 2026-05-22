#!/usr/bin/env python3
"""Command entrypoint for Exp 2843 BEAVER/EPR bounded-prefix probe."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.beaver_epr_bounded_probe import main


if __name__ == "__main__":
    raise SystemExit(main())
