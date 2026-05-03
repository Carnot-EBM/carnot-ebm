#!/usr/bin/env python3
"""Exp 1176: validate k=6 AND-composition with SC-Energy.

Spec: REQ-VERIFY-1176, SCENARIO-VERIFY-1176
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.k6_and_compose_validation import run_experiment  # noqa: E402


def main() -> None:
    artifact = run_experiment(project_root=PROJECT_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
