#!/usr/bin/env python3
"""Entrypoint for Exp 4314 IR3DE+CASCAL cross-domain selector rerun."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4314_cross_domain_selector_ir3de_cascal import main  # noqa: E402


if __name__ == "__main__":
    main()
