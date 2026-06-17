#!/usr/bin/env python3
"""Entrypoint for Exp 4331 learned frame encoder ARC transfer."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4331_self_learning_learned_frame_encoder_cross_game_transfer import main  # noqa: E402


if __name__ == "__main__":
    main()
