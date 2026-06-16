#!/usr/bin/env python3
"""Entrypoint for Exp 4296 ARC-AGI-3 different-game incremental progress."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from carnot.experiment_4296_arc_incremental_progress_new_game import main  # noqa: E402

if __name__ == "__main__":
    main()
