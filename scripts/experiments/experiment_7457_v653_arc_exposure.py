#!/usr/bin/env python3
"""Thin repository entrypoint for the V653 ARC exposure comparison."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from carnot.experiment_7457_v653_arc_exposure import main


if __name__ == "__main__":
    raise SystemExit(main())
