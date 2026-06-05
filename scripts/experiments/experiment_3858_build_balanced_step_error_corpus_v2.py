#!/usr/bin/env python3
"""Run Exp 3858 balanced step-error corpus v2 build."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.data.step_error_balanced_v2 import BuildConfig, write_corpus_artifact


if __name__ == "__main__":  # pragma: no cover
    artifact = write_corpus_artifact(BuildConfig(repo_root=REPO_ROOT))
    print(f"{REPO_ROOT / 'data' / 'step_error_balanced_v2.json'}")
    print(artifact["honest_verdict"])
