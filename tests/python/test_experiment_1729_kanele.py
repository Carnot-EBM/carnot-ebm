"""Tests for Exp 1729 KANELÉ verifier runner.

Spec: REQ-KAN-1729, SCENARIO-KAN-1729.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_1729_kanele as exp

def test_scenario_kan_1729_experiment_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-KAN-1729: the runner writes the completed KANELÉ artifact."""
    output_path = tmp_path / "experiment_1729_kanele.json"
    artifact = exp.run_experiment(
        output_path=str(output_path),
        run_date="20260510",
    )
    
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    
    assert persisted == artifact
    assert artifact["schema"] == "carnot.kanele.experiment_1729.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1729
    assert artifact["spec_traces"] == ["REQ-KAN-1729", "SCENARIO-KAN-1729"]

def test_main(tmp_path: Path) -> None:
    """Test CLI entrypoint."""
    cli_path = tmp_path / "cli_experiment_1729_kanele.json"
    rc = exp.main(["--output", str(cli_path), "--run-date", "20260510"])
    assert rc == 0
    assert json.loads(cli_path.read_text(encoding="utf-8"))["status"] == "complete"
