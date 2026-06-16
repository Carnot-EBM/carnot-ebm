#!/usr/bin/env python3
"""Entrypoint for Exp 4294 hardened verifier efficiency versus strong judges."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.verifier_efficiency_harden_strong_judge_4294 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
