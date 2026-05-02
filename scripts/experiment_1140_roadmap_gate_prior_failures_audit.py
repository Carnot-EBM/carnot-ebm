#!/usr/bin/env python3
"""Run Exp 1140 and write the roadmap audit deliverable.

Spec: REQ-INFRA-075
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from audit_roadmap_gates import audit_roadmap, select_roadmap_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUESTED_ROADMAP = PROJECT_ROOT / "research-roadmap-next.yaml"
COMPLETE_PATH = PROJECT_ROOT / "research-complete.yaml"
OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment_1140_roadmap_gate_prior_failures_audit.json"


def run_experiment(
    project_root: Path = PROJECT_ROOT,
    requested_roadmap: Path | None = None,
    complete_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Audit the requested roadmap and write the Exp 1140 JSON artifact."""
    requested = requested_roadmap or project_root / "research-roadmap-next.yaml"
    complete = complete_path or project_root / "research-complete.yaml"
    output = (
        output_path
        or project_root / "results" / "experiment_1140_roadmap_gate_prior_failures_audit.json"
    )
    active = project_root / "research-roadmap.yaml"
    roadmap_path, note = select_roadmap_path(requested, active_path=active)

    result = audit_roadmap(roadmap_path, complete_path=complete)
    artifact = result.to_artifact()
    artifact.update(
        {
            "experiment": "exp1140-roadmap-gate-prior-failures-audit",
            "schema_version": 1,
            "roadmap_path_requested": str(requested),
            "roadmap_path_used": str(roadmap_path),
            "roadmap_path_note": note,
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
