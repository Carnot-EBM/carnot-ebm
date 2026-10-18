#!/usr/bin/env python3
"""Thin repository entrypoint for the Exp7431 ARC live sentinel."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from carnot.experiment_7431_v651_arc_live_sentinel import main


if __name__ == "__main__":
    raise SystemExit(main())
