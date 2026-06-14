#!/usr/bin/env python3
"""Run Exp 4186 verifier efficiency moat versus LLM-as-judge.

Spec refs: REQ-VERIFY-4186, SCENARIO-VERIFY-4186.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from carnot.reporting.efficiency_moat_verifier_vs_llm_judge_4186 import run  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run(REPO_ROOT), indent=2, sort_keys=True))
