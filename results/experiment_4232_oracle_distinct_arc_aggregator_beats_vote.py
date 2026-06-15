#!/usr/bin/env python3
"""Run Exp 4232 held-out oracle-distinct ARC aggregator beats-vote gate.

Spec refs: REQ-VERIFY-4232, SCENARIO-VERIFY-4232,
SCENARIO-VERIFY-4232-NO-HEADROOM, SCENARIO-VERIFY-4232-DEFERRED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.oracle_distinct_arc_aggregator_4232 import run  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run(REPO_ROOT), indent=2, sort_keys=True))
