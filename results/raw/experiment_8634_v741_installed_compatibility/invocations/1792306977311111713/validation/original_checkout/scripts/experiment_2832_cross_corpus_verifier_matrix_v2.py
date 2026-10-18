#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.cross_corpus_verifier_matrix_v2 import run_analysis


if __name__ == "__main__":  # pragma: no cover
    run_analysis(REPO_ROOT)
