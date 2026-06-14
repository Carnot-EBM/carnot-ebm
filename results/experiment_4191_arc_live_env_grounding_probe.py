#!/usr/bin/env python3
"""Entrypoint for Exp 4191 ARC-AGI-3 live-env grounding probe."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4191_arc_live_env_grounding_probe import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    main()
