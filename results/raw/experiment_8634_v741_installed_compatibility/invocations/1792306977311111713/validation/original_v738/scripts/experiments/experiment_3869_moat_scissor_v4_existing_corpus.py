#!/usr/bin/env python3
"""Run Exp 3869 existing-corpus moat scissor v4."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.moat_scissor_v4_existing_corpus import ExperimentConfig, run_experiment


if __name__ == "__main__":  # pragma: no cover
    run_experiment(ExperimentConfig(repo_root=REPO_ROOT))
