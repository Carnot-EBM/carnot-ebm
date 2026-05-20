#!/usr/bin/env python3
"""Run Exp 2567 capstone: milestone .246 synthesis.

Spec: REQ-PUBLISH-031, SCENARIO-PUBLISH-031.

Reads the eleven .246 artifacts (exp2556-exp2566) plus the .246 roadmap
proposal, then writes the terminal deliverable to
results/experiment_2567_capstone_v246.json.

Missing artifacts are tracked as preconditions_checked entries and -- if
more than three -- surface as an EXECUTION_LAYER_GAP process flag so
the next planner can see the .246 execution shortfall.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.capstone_v246_2567 import write_artifact


def main() -> int:
    written = write_artifact(REPO_ROOT)
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
