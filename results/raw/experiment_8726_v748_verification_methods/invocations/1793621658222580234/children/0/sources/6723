#!/usr/bin/env python3
"""Experiment 1319: KAN hardware complexity audit.

Spec refs: REQ-KAN-1319, SCENARIO-KAN-1319.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from carnot.hardware import kan_hardware_complexity_audit as audit  # noqa: E402

EXP1148_PATH = audit.EXP1148_PATH
EXP1162_PATH = audit.EXP1162_PATH
EXP1174_PATH = audit.EXP1174_PATH
DELIVERABLE = audit.DELIVERABLE_PATH


def main() -> int:
    """Write the Exp 1319 hardware-portability audit deliverable."""
    artifact = audit.run_experiment(
        exp1148_path=EXP1148_PATH,
        exp1162_path=EXP1162_PATH,
        exp1174_path=EXP1174_PATH,
        deliverable_path=DELIVERABLE,
    )
    print(f"RM per inference       : {artifact['rm_per_inference']}")
    print(f"Lookup table bytes     : {artifact['lookup_table_bytes']}")
    print(f"Hardware claim allowed : {artifact['hardware_claim_allowed']}")
    print(f"Honest verdict         : {artifact['honest_verdict']}")
    print(f"Deliverable            : {DELIVERABLE}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
