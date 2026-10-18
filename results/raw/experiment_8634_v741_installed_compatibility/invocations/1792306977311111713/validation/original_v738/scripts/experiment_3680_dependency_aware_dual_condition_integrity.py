#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.dependency_aware_dual_condition_integrity import write_artifact


if __name__ == "__main__":  # pragma: no cover
    write_artifact(REPO_ROOT)
