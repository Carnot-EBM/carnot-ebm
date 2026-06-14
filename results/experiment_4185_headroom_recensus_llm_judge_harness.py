#!/usr/bin/env python3
"""Run Exp 4185 headroom re-census and LLM-as-judge harness.

Spec refs: REQ-VERIFY-4185, SCENARIO-VERIFY-4185.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
if (
    VENV_PYTHON.exists()
    and Path(sys.executable).resolve() != VENV_PYTHON.resolve()
    and os.environ.get("EXP4185_REEXEC") != "1"
):
    env = dict(os.environ)
    env["EXP4185_REEXEC"] = "1"
    os.execve(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve())], env)

sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from carnot.reporting.headroom_recensus_llm_judge_harness_4185 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
