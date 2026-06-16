#!/usr/bin/env python3
"""Entrypoint for Exp 4284 verifier efficiency versus LLM-as-judge."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.verifier_efficiency_vs_llm_judge_4284 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
