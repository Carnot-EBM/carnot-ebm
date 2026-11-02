#!/usr/bin/env python3
"""Run Exp 2580 capstone: milestone .247 synthesis.

Spec: REQ-PUBLISH-031, SCENARIO-PUBLISH-031.

Reads the eleven .247 artifacts (exp2569-exp2579) plus the .247 roadmap
proposal, then writes the terminal deliverable to
results/experiment_2580_capstone_v247.json.

Missing artifacts are tracked as preconditions_checked entries and -- if
more than four -- surface as an EXECUTION_LAYER_GAP process flag so the
next planner can see the .247 execution shortfall.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.capstone_v247_2580 import write_artifact


def main() -> int:
    written = write_artifact(REPO_ROOT)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
