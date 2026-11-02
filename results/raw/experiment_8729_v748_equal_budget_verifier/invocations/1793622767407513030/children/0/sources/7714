#!/usr/bin/env python3
"""Run Exp 3722 convergence synthesis.

Spec refs: REQ-REPORT-3722, SCENARIO-REPORT-3722-SYNTHESIZED,
SCENARIO-REPORT-3722-CANNOT-SYNTHESIZE.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.reporting.convergence_synthesis_operator_next_thesis_3722 import (
    write_artifact,
)  # noqa: E402


def main() -> int:
    output = write_artifact(REPO_ROOT)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
