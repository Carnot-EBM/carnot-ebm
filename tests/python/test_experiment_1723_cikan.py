"""Tests for Exp 1723 CIKAN verifier runner.

Spec: REQ-KAN-1723, SCENARIO-KAN-1723.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import experiment_1723_cikan as exp


def test_scenario_kan_1723_experiment_writes_required_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-KAN-1723: the runner writes the completed CIKAN artifact."""

    output_path = tmp_path / "experiment_1723_cikan.json"
    artifact = exp.run_experiment(
        output_path=output_path,
        run_date="20260510",
        epochs=40,
        tests_run=["test_scenario_kan_1723_experiment_writes_required_artifact"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["schema"] == "carnot.cikan.experiment_1723.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1723
    assert artifact["spec_traces"] == ["REQ-KAN-1723", "SCENARIO-KAN-1723"]
    assert artifact["fixed_boundaries_preserved"] is True
    assert artifact["constraint"]["expression"] == "X AND Y"
    assert artifact["metrics"]["accuracy"] == 1.0
    assert artifact["metrics"]["energy_gap"] > 0.0
    assert artifact["metrics"]["violating_energy"] > artifact["metrics"]["satisfying_energy"]
    assert artifact["honest_verdict"].startswith("complete:")

    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact({"status": "complete"})
    with pytest.raises(AssertionError, match="fixed_boundaries_preserved"):
        exp.validate_artifact(dict(artifact, fixed_boundaries_preserved=False))
    with pytest.raises(AssertionError, match="energy ordering"):
        exp.validate_artifact(
            dict(
                artifact,
                metrics=dict(
                    artifact["metrics"],
                    violating_energy=artifact["metrics"]["satisfying_energy"],
                ),
            )
        )

    cli_path = tmp_path / "cli_experiment_1723_cikan.json"
    rc = exp.main(["--output", str(cli_path), "--run-date", "20260510", "--epochs", "10"])
    assert rc == 0
    assert json.loads(cli_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert "wrote=" in capsys.readouterr().out


def test_req_kan_1723_extractor_env_restore_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-KAN-1723: toy FourierCSP extraction restores env and fails honestly."""

    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
    constraint = exp.extract_toy_constraint()
    assert constraint.expression == "X AND Y"
    assert "CARNOT_FORCE_LIVE" not in os.environ

    class EmptyExtractor:
        def __init__(self, generate_fn):  # noqa: ANN001
            self.generate_fn = generate_fn

        def extract(self, prompt: str):  # noqa: ANN201
            return None

    monkeypatch.setenv("CARNOT_FORCE_LIVE", "preserve")
    monkeypatch.setattr(exp, "FourierCSPExtractor", EmptyExtractor)
    with pytest.raises(RuntimeError, match="mock extraction failed"):
        exp.extract_toy_constraint()
    assert os.environ["CARNOT_FORCE_LIVE"] == "preserve"
