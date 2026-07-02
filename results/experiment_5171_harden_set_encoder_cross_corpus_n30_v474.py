#!/usr/bin/env python3
"""Entrypoint for Exp 5171 Set-Encoder cross-corpus n>=30 hardening."""

# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.harden_set_encoder_cross_corpus_n30_5171 import main  # noqa: E402


if __name__ == "__main__":
    main()
