#!/usr/bin/env python3
"""Run Exp 1400 BiPRM R2L retrospective FoVer pivot probe.

Spec: REQ-VERIFY-1400, SCENARIO-VERIFY-1400
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.biprm_retrospective_verification_probe import (  # noqa: E402
    DEFAULT_OUTPUT_PATH,
    RUN_DATE,
    run_experiment,
)


def main() -> int:
    """Write the Exp 1400 deliverable JSON."""

    artifact = run_experiment(output_path=DEFAULT_OUTPUT_PATH, limit=100, run_date=RUN_DATE)
    print(
        artifact.get("pivot_precision_delta"),
        artifact.get("retrospective_verification_viable"),
        artifact.get("honest_verdict"),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
