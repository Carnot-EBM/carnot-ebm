#!/usr/bin/env python3
"""Entrypoint for Exp 4295 fixed Tier-2 self-learning retrieval run."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.self_learning_tier2_fixed_retrieval_4295 import main  # noqa: E402


if __name__ == "__main__":
    main()
