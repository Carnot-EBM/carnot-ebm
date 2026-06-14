#!/usr/bin/env python3
"""Entrypoint for Exp 4190 ARC-AGI-3 hardened incremental progress."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4190_arc_incremental_progress import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    main()
