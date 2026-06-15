#!/usr/bin/env python3
"""Entrypoint for Exp 4270 ARC family provenance recovery."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _main() -> None:
    sys.path.insert(0, str(REPO_ROOT / "python"))
    from carnot.reporting.arc_family_provenance_recovery_4270 import main

    main()


if __name__ == "__main__":
    _main()
