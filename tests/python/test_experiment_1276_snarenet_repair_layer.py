"""Tests for Exp 1276 SnareNet-style adaptive repair on ContinuousEBM latents.

Spec refs: REQ-KONA-028, SCENARIO-KONA-028.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.phase3.continuous_ebm import AdaptiveRepairLayer
from scripts import experiment_1276_snarenet_repair_layer_gated as experiment


class TestAdaptiveRepairLayer:
    """REQ-KONA-028: adaptive repair continues from FSNet and reports diagnostics."""

    def test_improves_or_matches_fsnet_constraint_satisfaction(self) -> None:
        """REQ-KONA-028: appended adaptive repair improves the FSNet soft score."""
        state = np.array([0.82, 0.72, -0.25], dtype=np.float64)
        constraints = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        bias = np.array([-0.10, -0.08], dtype=np.float64)
        layer = AdaptiveRepairLayer(
            fsnet_steps=1,
            fsnet_lr=0.30,
            fsnet_anchor_weight=0.0,
            n_steps=12,
            lr=0.35,
            anchor_weight=0.0,
            initial_relaxation=0.20,
            min_relaxation=0.04,
            max_relaxation=0.50,
            tolerance=1e-10,
        )

        result = layer.repair(state, constraints, bias)

        assert result.initial_constraint_satisfaction < result.fsnet_constraint_satisfaction
        assert result.final_constraint_satisfaction > result.fsnet_constraint_satisfaction
        assert result.violation_energy <= result.fsnet_violation_energy
        assert result.repair_iterations > 0
        assert result.converged is True
        assert result.state.shape == state.shape
        assert np.all(result.state > -1.0)
        assert np.all(result.state < 1.0)
        assert result.distortion_from_initial >= result.fsnet_distortion_from_initial
        assert result.final_relaxation >= layer.min_relaxation

    def test_zero_adaptive_steps_returns_fsnet_state(self) -> None:
        """SCENARIO-KONA-028: zero adaptive iterations is exactly the FSNet baseline."""
        state = np.array([0.55, -0.2], dtype=np.float64)
        constraints = np.array([[1.0, 0.0]], dtype=np.float64)
        bias = np.array([-0.05], dtype=np.float64)
        layer = AdaptiveRepairLayer(
            fsnet_steps=4,
            fsnet_lr=0.35,
            n_steps=0,
            lr=0.30,
            tolerance=1e-9,
        )

        result = layer.repair(state, constraints, bias)

        np.testing.assert_allclose(result.state, result.fsnet_state)
        assert result.repair_iterations == 0
        assert result.final_constraint_satisfaction == pytest.approx(
            result.fsnet_constraint_satisfaction
        )
        assert result.distortion_from_fsnet == pytest.approx(0.0)

    def test_invalid_shapes_raise(self) -> None:
        """REQ-KONA-028: malformed latent or constraint shapes are rejected."""
        layer = AdaptiveRepairLayer()

        with pytest.raises(ValueError, match="state must be one-dimensional"):
            layer.repair(np.zeros((1, 2)), np.eye(2))

        with pytest.raises(ValueError, match="constraint_matrix"):
            layer.repair(np.zeros(2), np.ones(2))

        with pytest.raises(ValueError, match="constraint_bias"):
            layer.repair(np.zeros(2), np.eye(2), np.zeros(3))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"fsnet_steps": -1}, "fsnet_steps"),
            ({"fsnet_lr": -0.1}, "fsnet_lr"),
            ({"fsnet_anchor_weight": -0.1}, "fsnet_anchor_weight"),
            ({"n_steps": -1}, "n_steps"),
            ({"lr": -0.1}, "lr"),
            ({"anchor_weight": -0.1}, "anchor_weight"),
            ({"initial_relaxation": 0.0}, "initial_relaxation"),
            ({"min_relaxation": 0.0}, "min_relaxation"),
            ({"max_relaxation": 0.0}, "max_relaxation"),
            ({"relaxation_growth": 1.0}, "relaxation_growth"),
            ({"relaxation_decay": 1.0}, "relaxation_decay"),
            ({"tolerance": -1e-9}, "tolerance"),
        ],
    )
    def test_invalid_hyperparameters_raise(
        self,
        kwargs: dict[str, float | int],
        message: str,
    ) -> None:
        """REQ-KONA-028: invalid adaptive repair hyperparameters fail fast."""
        with pytest.raises(ValueError, match=message):
            AdaptiveRepairLayer(**kwargs)

    def test_relaxation_bounds_are_ordered(self) -> None:
        """REQ-KONA-028: min relaxation must not exceed max relaxation."""
        with pytest.raises(ValueError, match="min_relaxation"):
            AdaptiveRepairLayer(min_relaxation=0.5, max_relaxation=0.2)


def test_build_artifact_contains_required_exp1276_fields() -> None:
    """SCENARIO-KONA-028: artifact compares raw, FSNet, and adaptive repair arms."""
    artifact = experiment.build_artifact()

    assert artifact["schema"] == "carnot.phase3.snarenet_repair_layer.v1"
    assert artifact["experiment"] == "1276_snarenet_repair_layer_gated"
    assert artifact["run_date"] == "20260504"
    assert artifact["status"] == "complete"
    assert artifact["spec_refs"] == ["REQ-KONA-028", "SCENARIO-KONA-028"]
    assert artifact["source_context"]["experiment_1275_feasibility_delta_overall"] > 0.0
    assert set(artifact["arms"]) == {"raw_langevin", "fsnet_feasibility_step", "adaptive_repair"}
    assert artifact["final_constraint_satisfaction"] == pytest.approx(
        artifact["arms"]["adaptive_repair"]["constraint_satisfaction_mean"]
    )
    assert artifact["repair_iterations"] == pytest.approx(
        artifact["arms"]["adaptive_repair"]["repair_iterations_mean"]
    )
    assert artifact["distortion_from_initial"] == pytest.approx(
        artifact["arms"]["adaptive_repair"]["distortion_from_initial_mean"]
    )
    assert isinstance(artifact["diversity_preserved"], bool)
    assert artifact["repair_delta_over_fsnet"] >= 0.0
    assert artifact["honest_verdict"] in {
        "adaptive_repair_improves_fsnet",
        "adaptive_repair_matches_fsnet",
        "adaptive_repair_distorts_or_collapses",
        "blocked_exp1275_not_positive",
    }
    assert len(artifact["per_seed"]) == artifact["n_states"]
    json.dumps(artifact)


def test_script_main_writes_terminal_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-KONA-028: experiment script writes the terminal JSON artifact."""
    output_path = tmp_path / "experiment_1276_snarenet_repair_layer_gated.json"
    monkeypatch.setattr(experiment, "RESULT_PATH", output_path)

    artifact = experiment.main()

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written == artifact
    assert written["status"] == "complete"
    assert "honest_verdict" in written
