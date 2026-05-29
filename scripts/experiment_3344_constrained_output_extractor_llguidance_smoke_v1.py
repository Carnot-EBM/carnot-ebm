#!/usr/bin/env python3
"""Run Exp 3344 Constrained Output Extractor llguidance Smoke v1."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.constrained_output_extractor_llguidance_smoke_v1_3344 import (  # noqa: E402
    run_experiment,
)


def main() -> int:
    artifact = run_experiment(project_root=REPO_ROOT)
    
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "experiment_3344_constrained_output_extractor_llguidance_smoke_v1.json"
    
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
        f.write("\n")
        
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
