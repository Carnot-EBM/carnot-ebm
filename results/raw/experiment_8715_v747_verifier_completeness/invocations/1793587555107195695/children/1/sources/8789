#!/usr/bin/env python3
"""Repository entry point for REQ-VERIFY-6799."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6799_model_output_formal_constraint_probes import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
