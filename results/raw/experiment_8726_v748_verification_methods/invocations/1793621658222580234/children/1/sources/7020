#!/usr/bin/env python3
"""Run Exp 2481 capstone: paper-v6 synthesis for milestone 2026.05.239.

Spec: REQ-REPORT-2481, SCENARIO-REPORT-2481.

Reads all .239 artifacts, computes headline metrics (best AUROC,
gap-to-HIVE, hardware status, Phase 4 hold status, arXiv readiness),
writes paper-v6 results-table fragment and a .239 update subsection
into ``docs/arxiv-paper/main.tex``, and emits the terminal deliverable
to ``results/experiment_2481_capstone_v239.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.paper_v6_capstone_2481 import main


if __name__ == "__main__":
    raise SystemExit(main())
