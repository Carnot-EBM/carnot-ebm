#!/usr/bin/env python3
"""Run Exp 1365 Eidoku CSP neuro-symbolic verification probe.

Spec: REQ-VERIFY-1365, SCENARIO-VERIFY-1365
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.eidoku_csp_probe import DEFAULT_OUTPUT_PATH, run_experiment  # noqa: E402


def main() -> int:
    """Write the Exp 1365 deliverable JSON."""

    artifact = run_experiment(output_path=DEFAULT_OUTPUT_PATH, limit=100)
    print(
        artifact.get("corpus_cases_used"),
        artifact.get("csp_feasibility_rate"),
        artifact.get("eidoku_csp_viable"),
        artifact.get("honest_verdict"),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
