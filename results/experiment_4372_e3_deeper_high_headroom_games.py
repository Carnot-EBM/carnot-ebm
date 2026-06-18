#!/usr/bin/env python3
"""Entry point wrapper for Exp 4372."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python"))

from carnot.experiment_4372_e3_deeper_high_headroom_games import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
