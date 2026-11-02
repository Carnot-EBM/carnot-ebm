#!/usr/bin/env python3
"""Command entrypoint for Exp 2866 tiny exact BEAVER frontier feasibility."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.beaver_exact_tiny_frontier import main


if __name__ == "__main__":
    raise SystemExit(main())
