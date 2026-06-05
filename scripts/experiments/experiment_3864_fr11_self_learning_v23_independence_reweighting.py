#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.fr11.continuous_self_learning_v23 import write_artifact


if __name__ == "__main__":  # pragma: no cover
    write_artifact(REPO_ROOT)
