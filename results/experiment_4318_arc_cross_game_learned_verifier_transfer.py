#!/usr/bin/env python3
"""Entrypoint for Exp 4318 ARC cross-game learned value-head transfer."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4318_arc_cross_game_learned_verifier_transfer import main  # noqa: E402


if __name__ == "__main__":
    main()
