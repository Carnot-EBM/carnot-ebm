"""Tests for the Exp 1709 `.131` archive and `.132` initialization artifact.

Spec: REQ-REPORT-131, SCENARIO-REPORT-131.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import milestone_131_archive as exp


def test_req_report_131_run_writes_terminal_json(tmp_path: Path) -> None:
    """REQ-REPORT-131: run writes terminal archive JSON."""
    output_path = tmp_path / "ops" / "lineage-retirements" / "milestone_131_archive.json"

    artifact = exp.run(root=tmp_path, output_path=output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["status"] == "complete"
    assert written["milestone"] == "2026.05.132"
    assert exp.REQUIRED_ARTIFACT_FIELDS <= set(artifact)


def test_build_artifact() -> None:
    """Test build artifact fields for REQ-REPORT-131."""
    artifact = exp.build_artifact()
    assert artifact["experiment"] == exp.EXPERIMENT
    assert artifact["status"] == "complete"
