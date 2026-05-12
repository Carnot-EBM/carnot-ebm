"""Tests for the Exp 1961 IGD benchmark artifact.

Spec traces: REQ-IGD-1961-5, SCENARIO-IGD-1961.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.experiment_1961_interleaved_gibbs_diffusion as exp


def test_scenario_igd_1961_experiment_writes_required_results_artifact(tmp_path: Path) -> None:
    """REQ-IGD-1961-5: runner writes the required terminal JSON artifact."""
    output_path = tmp_path / "experiment_1961_interleaved_gibbs_diffusion.json"

    artifact = exp.run_experiment(output_path=output_path)
    written = json.loads(output_path.read_text())

    assert written == artifact
    assert artifact["artifact_path"] == str(output_path)
    assert artifact["experiment_id"] == "1961"
    assert artifact["status"] == "complete"
    assert artifact["run_date"] == "20260512"
    assert "REQ-IGD-1961-5" in artifact["spec_refs"]
    assert artifact["problem"]["num_variables"] > 0
    assert artifact["problem"]["num_clauses"] > 0
    assert artifact["samplers"]["igd"]["uses_continuous_noise"] is True
    assert artifact["samplers"]["sequential_gibbs"]["uses_continuous_noise"] is False
    assert artifact["metrics"]["igd"]["best_satisfied"] <= artifact["problem"]["num_clauses"]
    assert artifact["metrics"]["sequential_gibbs"]["best_satisfied"] <= artifact["problem"]["num_clauses"]
