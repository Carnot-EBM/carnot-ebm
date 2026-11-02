#!/usr/bin/env python3
"""Run Exp 3069 solver-verifier failure autopsy protocol."""

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

from carnot.reporting.solver_verifier_failure_autopsy_protocol_3069 import (  # noqa: E402
    OUTPUT_REL_PATH,
    build_artifact,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


def main() -> int:
    template = ExperimentTemplate(
        exp_id=3069,
        title="Solver-verifier failure autopsy protocol",
        deliverable=OUTPUT_REL_PATH.as_posix(),
        requires_gpu=False,
        repo_root=REPO_ROOT,
        seed=3069,
    )
    template.setup()
    artifact = build_artifact(REPO_ROOT)
    output_path = REPO_ROOT / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    template.assert_deliverable_written()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["verifier_failure_autopsy_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
