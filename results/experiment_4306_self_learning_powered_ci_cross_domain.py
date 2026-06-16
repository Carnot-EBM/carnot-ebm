#!/usr/bin/env python3
"""Entrypoint for Exp 4306 powered cross-domain self-learning CI."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4306_self_learning_powered_ci_cross_domain import main  # noqa: E402


if __name__ == "__main__":
    main()
