#!/usr/bin/env python3
"""Run Exp 4264 code oracle-distinct replication retry.

Spec refs: REQ-VERIFY-4264, SCENARIO-VERIFY-4264.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "python"))
    module = importlib.import_module("carnot.reporting.code_oracle_distinct_replication_retry_4264")
    run = module.run
    print(json.dumps(run(REPO_ROOT), indent=2, sort_keys=True))
