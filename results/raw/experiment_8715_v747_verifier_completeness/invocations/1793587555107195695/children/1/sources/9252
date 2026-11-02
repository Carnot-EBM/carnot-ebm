#!/usr/bin/env python3
"""Thin entrypoint for the V654 ARC decision-seam observation pilot."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from carnot.experiment_7471_v654_arc_seam_observation import main


if __name__ == "__main__":
    raise SystemExit(main())
