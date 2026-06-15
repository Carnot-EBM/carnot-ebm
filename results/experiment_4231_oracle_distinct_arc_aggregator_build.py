#!/usr/bin/env python3
"""Run Exp 4231 cross-candidate oracle-distinct ARC aggregator build.

Spec refs: REQ-VERIFY-4231, SCENARIO-VERIFY-4231,
SCENARIO-VERIFY-4231-NO-GAIN, SCENARIO-VERIFY-4231-BLOCKED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.oracle_distinct_arc_aggregator_4231 import run  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run(REPO_ROOT), indent=2, sort_keys=True))
