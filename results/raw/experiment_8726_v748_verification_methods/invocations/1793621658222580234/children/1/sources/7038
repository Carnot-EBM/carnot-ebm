#!/usr/bin/env python3
"""Run Exp 2516 capstone: paper-v6 synthesis for milestone 2026.05.242.

Spec: REQ-REPORT-2516, SCENARIO-REPORT-2516.

Reads all .242 artifacts (exp2507-2515), computes the four
arXiv-readiness gates (Phase 1 ship, audit, Phase 4 step-level
validation, AUROC adversarial verification), surfaces methodology
fallbacks via corrigendum_pending, and writes the terminal deliverable
to results/experiment_2516_capstone_v242.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.paper_v6_capstone_2516 import main


if __name__ == "__main__":
    raise SystemExit(main())
