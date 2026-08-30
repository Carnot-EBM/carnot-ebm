#!/usr/bin/env python3
"""Repository entry point for REQ-VERIFY-6788."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_6788_soft_fixed_point_structural_control_ab import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
