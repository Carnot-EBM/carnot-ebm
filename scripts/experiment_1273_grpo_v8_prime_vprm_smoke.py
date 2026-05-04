#!/usr/bin/env python3
"""CLI wrapper for Exp 1273 GRPO v8 PRIME/VPRM smoke."""

from __future__ import annotations

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
if __name__ == "__main__" and VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.training.grpo_v8_prime_vprm_smoke import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
