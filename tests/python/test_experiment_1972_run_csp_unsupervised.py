"""Tests for Exp 1972 RUN-CSP unsupervised solver artifact.

Spec traces: REQ-SAMPLE-1972, SCENARIO-SAMPLE-1972.
"""

from __future__ import annotations

import json
from pathlib import Path

import scripts.experiment_1972_run_csp_unsupervised as exp


REQUIRED_FIELDS = {
    "status",
    "experiment_id",
    "spec_refs",
    "run_date",
    "solver_name",
    "paper_reference",
    "problem_family",
    "config",
    "training",
    "evaluations",
    "labels_used",
    "cpu_only",
    "hardware_execution_performed",
    "network_access_used",
    "artifact_path",
    "honest_verdict",
}


def test_scenario_sample_1972_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-1972: experiment writes terminal RUN-CSP JSON."""

    output_path = tmp_path / "experiment_1972_run_csp_unsupervised.json"
    artifact = exp.run_experiment(output_path=output_path)

    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1972
    assert artifact["spec_refs"] == ["REQ-SAMPLE-1972", "SCENARIO-SAMPLE-1972"]
    assert artifact["paper_reference"] == "arXiv:1909.08387"
    assert artifact["labels_used"] is False
    assert artifact["cpu_only"] is True
    assert artifact["hardware_execution_performed"] is False
    assert artifact["network_access_used"] is False
    assert artifact["artifact_path"] == str(output_path)
    assert artifact["training"]["graph"]["num_variables"] == 40
    assert "40" in artifact["evaluations"]
    assert "1000" in artifact["evaluations"]
    assert artifact["evaluations"]["1000"]["num_variables"] == 1000
    assert artifact["evaluations"]["1000"]["satisfaction_rate"] >= 0.95
    assert "cpu_only" in artifact["honest_verdict"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_scenario_sample_1972_main_prints_summary(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """SCENARIO-SAMPLE-1972: CLI entry point writes and reports the artifact."""

    output_path = tmp_path / "experiment_1972_run_csp_unsupervised.json"
    monkeypatch.setattr(exp, "DEFAULT_RESULT_PATH", output_path)

    exp.main()

    captured = capsys.readouterr().out
    assert "1972" in captured
    assert "1000" in captured
    assert output_path.exists()
