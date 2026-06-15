#!/usr/bin/env python3
"""Run Exp 4256 ARC oracle-distinct provenance leak audit.

Spec refs: REQ-VERIFY-4256, SCENARIO-VERIFY-4256.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.arc_oracle_distinct_leak_audit_4256 import run  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(run(REPO_ROOT), indent=2, sort_keys=True))
