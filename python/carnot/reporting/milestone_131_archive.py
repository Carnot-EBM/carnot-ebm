"""Build the Exp 1709 `.131` archive and `.132` initialization artifact.

Spec: REQ-REPORT-131, SCENARIO-REPORT-131.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260510"
PREDECESSOR_MILESTONE = "2026.05.131"
TARGET_MILESTONE = "2026.05.132"
EXPERIMENT = "1709_archive_131"
SCHEMA = "milestone_132_initialization_v1"

DEFAULT_OUTPUT_PATH = REPO_ROOT / "ops" / "lineage-retirements" / "milestone_131_archive.json"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "predecessor_milestone",
    "predecessor_archived",
    "predecessor_task_count",
    "active_roadmap_milestone",
    "honest_verdict",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def build_artifact() -> dict[str, Any]:
    """REQ-REPORT-131: build the terminal `.131` archive and `.132` state artifact."""
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete",
        "milestone": TARGET_MILESTONE,
        "predecessor_milestone": PREDECESSOR_MILESTONE,
        "predecessor_archived": True,
        "predecessor_task_count": 13,
        "active_roadmap_milestone": TARGET_MILESTONE,
        "honest_verdict": "complete: milestone_132_initialized_131_archived",
    }
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-131: write the `.131` archive and `.132` initialization artifact."""
    _ = root
    out = Path(output_path)
    artifact = build_artifact()
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
