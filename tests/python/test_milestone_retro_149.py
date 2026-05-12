"""Tests for the Exp 1917 milestone .149 retrospective.

Spec: REQ-REPORT-1917, SCENARIO-REPORT-1917.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_149 import REQUIRED_ARTIFACT_FIELDS, run


def test_scenario_report_1917_generates_retro(tmp_path: Path) -> None:
    """SCENARIO-REPORT-1917: Verify retro JSON generation for .149."""
    out = tmp_path / "retro.json"
    artifact = run(root=tmp_path, out_path=out, tests_run=["pytest passed"])
    
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone_149_retro_complete"] is True
    assert artifact["completed_task_count"] == 2
    assert artifact["blocked_task_count"] == 9
    assert artifact["failed_task_count"] == 2
    assert "pytest passed" in artifact["tests_run"]

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == artifact

def test_req_report_1917_default_tests_run_is_empty(tmp_path: Path) -> None:
    """REQ-REPORT-1917: Verify default tests_run is empty."""
    out = tmp_path / "retro.json"
    artifact = run(root=tmp_path, out_path=out)
    assert artifact["tests_run"] == []

