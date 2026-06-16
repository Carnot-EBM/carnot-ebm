#!/usr/bin/env python3
"""Entrypoint for Exp 4283 repowered ARC self-learning adaptation."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.self_learning_repowered_arcgen_4283 import main  # noqa: E402


if __name__ == "__main__":
    main()
