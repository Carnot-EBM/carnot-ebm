#!/usr/bin/env python3
"""Runner for Exp 3348 Independent Reproducer Pack and Evidence Matrix v40."""

from __future__ import annotations

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_3348_independent_reproducer_pack_evidence_matrix_v40 import run_experiment

def main() -> int:
    artifact = run_experiment(project_root=REPO_ROOT)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    sys.exit(main())
