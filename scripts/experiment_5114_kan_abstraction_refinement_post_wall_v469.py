#!/usr/bin/env python
"""CLI wrapper for Exp 5114 KAN abstraction-refinement post-wall diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.experiment_5114_kan_abstraction_refinement_post_wall_v469 import main


if __name__ == "__main__":
    raise SystemExit(main())
