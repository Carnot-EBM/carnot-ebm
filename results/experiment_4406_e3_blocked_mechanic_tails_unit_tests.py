#!/usr/bin/env python3
"""Entry point wrapper for Exp 4406.

Spec refs: REQ-PHASE4-4406, SCENARIO-PHASE4-4406.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "python"))

from carnot.experiment_4406_e3_blocked_mechanic_tails_unit_tests import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
