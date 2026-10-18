#!/usr/bin/env python3
"""Run Exp 3002 deterministic metamorphic repair-oracle audit.

Spec: REQ-CODE-3002, SCENARIO-CODE-3002.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from carnot.eval.metamorphic_repair_oracle_audit import (  # noqa: E402
    ARTIFACT_FILENAME,
    ExperimentConfig,
    write_artifact,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402


def main() -> int:
    template = ExperimentTemplate(
        exp_id=3002,
        title="Metamorphic Hard-Set Repair Oracle Audit",
        deliverable=f"results/{ARTIFACT_FILENAME}",
        requires_gpu=False,
        seed=3002,
    )
    template.setup()
    artifact = write_artifact(
        ExperimentConfig(
            tests_run=(
                ".venv/bin/pytest tests/python/test_experiment_3002_metamorphic_repair_oracle_audit.py -q",
                ".venv/bin/pytest tests/python -q",
                "python scripts/check_spec_coverage.py",
            )
        )
    )
    template.assert_deliverable_written()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["metamorphic_oracle_ready"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
