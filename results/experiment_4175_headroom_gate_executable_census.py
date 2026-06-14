#!/usr/bin/env python3
"""Run Exp 4175 executable headroom gate census.

Spec refs: REQ-VERIFY-4175, SCENARIO-VERIFY-4175.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.headroom_gate import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main([str(REPO_ROOT)]))
