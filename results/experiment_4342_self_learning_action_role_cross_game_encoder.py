#!/usr/bin/env python3
"""Entrypoint for Exp 4342 action-role ARC cross-game encoder transfer."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4342_self_learning_action_role_cross_game_encoder import main  # noqa: E402


if __name__ == "__main__":
    main()
