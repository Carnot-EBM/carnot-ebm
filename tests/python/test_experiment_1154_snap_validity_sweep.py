"""Tests for Exp 1154 latent-to-validity snap sweep.

Spec coverage: REQ-KONA-008, SCENARIO-KONA-007
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.phase3.continuous_ebm import ContinuousEBM
from carnot.phase3.snap_validity import (
    SnapSweepConfig,
    build_snap_validity_artifact,
    build_synthetic_action_space,
    infer_latent_dim,
    sample_uniform_latents,
    snap_states_to_actions,
    snap_validity_verdict,
    snapped_actions_legal_mask,
    run_snap_validity_sweep,
)
from scripts import experiment_1154_snap_validity_sweep as experiment


def _model(latent_dim: int = 3) -> ContinuousEBM:
    return ContinuousEBM(
        variables=latent_dim,
        coupling=np.zeros((latent_dim, latent_dim), dtype=np.float64),
        bias=np.zeros(latent_dim, dtype=np.float64),
    )


class TestLatentDimension:
    """REQ-KONA-008: the sweep derives d from the Phase 3 ContinuousEBM."""

    def test_infer_latent_dim_uses_continuous_ebm_variables(self) -> None:
        """REQ-KONA-008: latent_dim equals ContinuousEBM.variables."""
        assert infer_latent_dim(_model(latent_dim=10)) == 10

    def test_infer_latent_dim_rejects_non_positive_dimensions(self) -> None:
        """REQ-KONA-008: invalid Phase 3 latent dimensions fail fast."""
        with pytest.raises(ValueError, match="positive"):
            infer_latent_dim(_model(latent_dim=0))


class TestSyntheticProxyActionSpace:
    """REQ-KONA-008: the fallback proxy is a capped 0.1-spaced legal grid."""

    def test_build_synthetic_action_space_caps_actions(self) -> None:
        """REQ-KONA-008: proxy action count is capped to the configured maximum."""
        actions = build_synthetic_action_space(latent_dim=10, spacing=0.1, max_actions=1_000)
        assert actions.shape == (1_000, 10)
        assert np.all(actions >= -1.0)
        assert np.all(actions <= 1.0)

    def test_build_synthetic_action_space_uses_spacing_grid(self) -> None:
        """REQ-KONA-008: every action coordinate lies on the 0.1-spaced grid."""
        actions = build_synthetic_action_space(latent_dim=2, spacing=0.1, max_actions=50)
        scaled = np.round((actions + 1.0) / 0.1)
        np.testing.assert_allclose(actions, -1.0 + scaled * 0.1, atol=1e-12)

    def test_build_synthetic_action_space_caps_to_full_grid_for_one_dim(self) -> None:
        """REQ-KONA-008: the proxy does not duplicate actions when the grid is small."""
        actions = build_synthetic_action_space(latent_dim=1, spacing=0.1, max_actions=1_000)
        assert actions.shape == (21, 1)
        assert len({tuple(row) for row in actions}) == 21

    @pytest.mark.parametrize(
        ("latent_dim", "spacing", "max_actions", "message"),
        [
            (0, 0.1, 10, "latent_dim"),
            (2, 0.0, 10, "spacing"),
            (2, 0.1, 0, "max_actions"),
        ],
    )
    def test_build_synthetic_action_space_validates_inputs(
        self,
        latent_dim: int,
        spacing: float,
        max_actions: int,
        message: str,
    ) -> None:
        """REQ-KONA-008: invalid proxy-grid parameters fail before sampling."""
        with pytest.raises(ValueError, match=message):
            build_synthetic_action_space(
                latent_dim=latent_dim,
                spacing=spacing,
                max_actions=max_actions,
            )


class TestSamplingAndSnap:
    """SCENARIO-KONA-007: continuous states snap to nearest legal actions."""

    def test_sample_uniform_latents_is_reproducible_and_bounded(self) -> None:
        """SCENARIO-KONA-007: samples are deterministic and inside [-1, 1]^d."""
        a = sample_uniform_latents(latent_dim=3, n_states=8, seed=1154)
        b = sample_uniform_latents(latent_dim=3, n_states=8, seed=1154)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (8, 3)
        assert np.all(a >= -1.0)
        assert np.all(a <= 1.0)

    def test_sample_uniform_latents_validates_inputs(self) -> None:
        """SCENARIO-KONA-007: invalid sample requests fail fast."""
        with pytest.raises(ValueError, match="n_states"):
            sample_uniform_latents(latent_dim=3, n_states=0, seed=1154)

    def test_snap_states_to_actions_finds_nearest_grid_point(self) -> None:
        """SCENARIO-KONA-007: snap uses Euclidean nearest-neighbor distance."""
        states = np.array([[0.91, -0.88], [-0.1, 0.18]], dtype=np.float64)
        actions = np.array([[1.0, -0.9], [0.0, 0.2], [-1.0, 1.0]], dtype=np.float64)
        snapped, distances = snap_states_to_actions(states, actions, chunk_size=1)
        np.testing.assert_array_equal(snapped, np.array([[1.0, -0.9], [0.0, 0.2]]))
        np.testing.assert_allclose(distances, [np.sqrt(0.0085), np.sqrt(0.0104)])

    def test_snap_states_to_actions_validates_shapes(self) -> None:
        """SCENARIO-KONA-007: state/action dimensionality mismatches are rejected."""
        with pytest.raises(ValueError, match="same latent dimension"):
            snap_states_to_actions(np.zeros((2, 3)), np.zeros((4, 2)))

    def test_snapped_actions_legal_mask_checks_action_set_membership(self) -> None:
        """SCENARIO-KONA-007: a snapped point is legal iff it exists in the legal set."""
        snapped = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype=np.float64)
        legal = np.array([[0.0, 0.1], [0.4, 0.5]], dtype=np.float64)
        mask = snapped_actions_legal_mask(snapped, legal)
        np.testing.assert_array_equal(mask, np.array([True, False, True]))


class TestArtifact:
    """REQ-KONA-008: artifact fields and verdict thresholds are canonical."""

    @pytest.mark.parametrize(
        ("rate", "expected"),
        [
            (0.95, "option_a_viable_above_95pct"),
            (0.9499, "option_a_marginal_90_to_95pct"),
            (0.9, "option_a_marginal_90_to_95pct"),
            (0.8999, "option_a_failed_below_90pct"),
        ],
    )
    def test_snap_validity_verdict_thresholds(self, rate: float, expected: str) -> None:
        """REQ-KONA-008: honest_verdict follows the 95% and 90% thresholds."""
        assert snap_validity_verdict(rate) == expected

    def test_snap_validity_verdict_handles_missing_continuous_ebm(self) -> None:
        """REQ-KONA-008: missing Phase 3 ContinuousEBM gets the required verdict."""
        assert (
            snap_validity_verdict(0.0, continuous_ebm_found=False)
            == "phase3_continuous_ebm_not_found"
        )

    def test_build_snap_validity_artifact_contains_required_fields(self) -> None:
        """REQ-KONA-008: artifact includes every required snap-sweep field."""
        artifact = build_snap_validity_artifact(
            latent_dim=10,
            n_states_sampled=100,
            n_legal_snaps=96,
            proxy_used=True,
            action_space_description="synthetic proxy",
            continuous_ebm_found=True,
        )
        assert artifact["latent_dim"] == 10
        assert artifact["n_states_sampled"] == 100
        assert artifact["n_legal_snaps"] == 96
        assert artifact["snap_validity_rate"] == pytest.approx(0.96)
        assert artifact["snap_validity_gate_passed"] is True
        assert artifact["phase4_option_a_viable"] is True
        assert artifact["proxy_used"] is True
        assert artifact["action_space_description"] == "synthetic proxy"
        assert artifact["honest_verdict"] == "option_a_viable_above_95pct"
        json.dumps(artifact)

    def test_build_snap_validity_artifact_requires_positive_sample_count(self) -> None:
        """REQ-KONA-008: malformed artifacts with zero samples are rejected."""
        with pytest.raises(ValueError, match="n_states_sampled"):
            build_snap_validity_artifact(
                latent_dim=10,
                n_states_sampled=0,
                n_legal_snaps=0,
                proxy_used=True,
                action_space_description="synthetic proxy",
            )


def test_run_snap_validity_sweep_uses_proxy_and_gate_identity() -> None:
    """SCENARIO-KONA-007: sweep returns a proxy artifact with matched gate fields."""
    artifact = run_snap_validity_sweep(
        model=_model(latent_dim=2),
        config=SnapSweepConfig(n_states=40, seed=1154, max_actions=25, chunk_size=7),
    )
    assert artifact["latent_dim"] == 2
    assert artifact["n_states_sampled"] == 40
    assert artifact["n_legal_snaps"] == 40
    assert artifact["snap_validity_rate"] == pytest.approx(1.0)
    assert artifact["snap_validity_gate_passed"] is artifact["phase4_option_a_viable"]
    assert artifact["proxy_used"] is True
    assert "0.1-spaced" in artifact["action_space_description"]


def test_script_main_writes_required_deliverable_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-KONA-007: experiment script writes the canonical JSON artifact."""
    output_path = tmp_path / "experiment_1154_snap_validity_sweep.json"
    monkeypatch.setattr(experiment, "RESULT_PATH", output_path)
    monkeypatch.setattr(
        experiment,
        "DEFAULT_CONFIG",
        SnapSweepConfig(n_states=12, seed=1154, max_actions=9, chunk_size=4),
    )

    artifact = experiment.main()

    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["n_states_sampled"] == 12
    assert written["phase4_option_a_viable"] is written["snap_validity_gate_passed"]
    assert written["honest_verdict"] == "option_a_viable_above_95pct"
