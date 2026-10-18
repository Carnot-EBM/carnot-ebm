#!/usr/bin/env python3
"""Run exact certification for the frozen Exp6956 mapping bank.

Spec ref: REQ-VERIFY-6957.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6957_smt_mapping_certification import main


if __name__ == "__main__":  # pragma: no cover - required command surface.
    raise SystemExit(main())
