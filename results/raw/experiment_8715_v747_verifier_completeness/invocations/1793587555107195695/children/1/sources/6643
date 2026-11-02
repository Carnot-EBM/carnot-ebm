#!/usr/bin/env python3
"""Experiment 1162: KANELE SOS-KAN FPGA LUT blueprint.

This runner generates the JSON deliverable and markdown hardware blueprint for
the compressed Exp 1148 SOSKANEnergyV3 shape.  It is intentionally synthesis
free because the task is to produce a quantized table specification and
hardware-oriented complexity estimate, not RTL.

Spec refs: REQ-KAN-1162, SCENARIO-KAN-1162.
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

from carnot.analysis import kanele_sos_kan_fpga as kanele  # noqa: E402

EXP1148_PATH = kanele.EXP1148_PATH
DELIVERABLE = kanele.DELIVERABLE_PATH
BLUEPRINT_PATH = kanele.BLUEPRINT_PATH


def main() -> int:
    """Write the Exp 1162 deliverable JSON and SOS-KAN LUT blueprint."""
    kanele.run_experiment(
        exp1148_path=EXP1148_PATH,
        deliverable_path=DELIVERABLE,
        blueprint_path=BLUEPRINT_PATH,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
