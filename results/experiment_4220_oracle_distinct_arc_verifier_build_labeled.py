#!/usr/bin/env python3
"""Run Exp 4220 labeled oracle-distinct ARC verifier build.

Spec refs: REQ-VERIFY-4220, SCENARIO-VERIFY-4220,
SCENARIO-VERIFY-4220-BLOCKED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.oracle_distinct_arc_verifier_4220 import run  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run(REPO_ROOT), indent=2, sort_keys=True))
