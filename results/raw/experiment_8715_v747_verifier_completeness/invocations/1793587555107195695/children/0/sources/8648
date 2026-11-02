#!/usr/bin/env python3
"""Exp 3935 entry point for the competent LLM judge build.

Spec refs: REQ-VERIFY-3925, SCENARIO-VERIFY-3925.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.experiment_3925_competent_judge_build import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
