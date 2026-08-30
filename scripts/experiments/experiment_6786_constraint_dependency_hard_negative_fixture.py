#!/usr/bin/env python3
"""Repository entry point for REQ-VERIFY-6786."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_6786_constraint_dependency_hard_negative_fixture import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
