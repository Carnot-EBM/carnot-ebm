#!/usr/bin/env python3
"""Run Exp 4246 second-corpus code oracle-distinct replication gate.

Spec refs: REQ-VERIFY-4246, SCENARIO-VERIFY-4246,
SCENARIO-VERIFY-4246-BLOCKED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.code_oracle_distinct_replication_4246 import run  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run(REPO_ROOT), indent=2, sort_keys=True))
