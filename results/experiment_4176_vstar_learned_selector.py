#!/usr/bin/env python3
"""Run Exp 4176 V-STaR learned selector.

Spec refs: REQ-VERIFY-4176, SCENARIO-VERIFY-4176.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.vstar_learned_selector_4176 import run  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run(REPO_ROOT), indent=2, sort_keys=True))
