#!/usr/bin/env python3
"""Entrypoint for Exp 4303 verifier efficiency parity iso-FLOPs rerun."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.verifier_efficiency_parity_isoflops_4303 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
