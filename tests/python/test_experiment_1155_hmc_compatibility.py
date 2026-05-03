"""Tests for Exp 1155 HMC compatibility diagnostics.

Spec coverage: REQ-KONA-009, SCENARIO-KONA-008
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.phase3.hmc_compatibility import (
    HMCCompatibilityConfig,
    LatentEnergyComponent,
    build_default_continuous_subspace_components,
    build_default_latent_components,
    build_hmc_compatibility_artifact,
    classify_d1_signal,
    classify_d2_signal,
    classify_d3_signal,
    classify_hmc_regime,
    finite_difference_gradient,
    gradient_disparity_ratio,
    load_latent_dim_from_exp1154,
    run_hmc_compatibility_diagnostics,
)
from scripts import experiment_1155_hmc_compatibility_diagnostics as experiment


def _quadratic_component(name: str, scale: float = 1.0) -> LatentEnergyComponent:
    return LatentEnergyComponent(
        name=name,
        energy_fn=lambda x: float(0.5 * scale * np.dot(x, x)),
        continuous=True,
    )


class TestThresholds:
    """REQ-KONA-009: D1-D3 thresholds match the Exp 1155 task prompt."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0099, "A"), (0.01, "B"), (0.0999, "B"), (0.1, "C")],
    )
    def test_d1_thresholds(self, value: float, expected: str) -> None:
        """REQ-KONA-009: D1 classifies reversibility error at 0.01 and 0.1."""
        assert classify_d1_signal(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.099, "A"), (0.1, "B"), (0.999, "B"), (1.0, "C")],
    )
    def test_d2_thresholds(self, value: float, expected: str) -> None:
        """REQ-KONA-009: D2 classifies Hamiltonian variance at 0.1 and 1.0."""
        assert classify_d2_signal(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(9.99, "A"), (10.0, "B"), (99.99, "B"), (100.0, "C")],
    )
    def test_d3_thresholds(self, value: float, expected: str) -> None:
        """REQ-KONA-009: D3 classifies gradient disparity at 10 and 100."""
        assert classify_d3_signal(value) == expected

    @pytest.mark.parametrize(
        ("signals", "expected"),
        [(("A", "A", "A"), "A"), (("A", "B", "A"), "B"), (("C", "A", "B"), "C")],
    )
    def test_hmc_regime_uses_worst_signal(
        self, signals: tuple[str, str, str], expected: str
    ) -> None:
        """REQ-KONA-009: final HMC regime is the worst D1-D3 signal."""
        assert classify_hmc_regime(*signals) == expected


def test_finite_difference_gradient_matches_quadratic() -> None:
    """SCENARIO-KONA-008: numerical_fd gradients approximate smooth energies."""
    x = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    grad = finite_difference_gradient(lambda z: float(0.5 * np.dot(z, z)), x, eps=1e-6)
    np.testing.assert_allclose(grad, x, rtol=1e-5, atol=1e-6)


def test_build_artifact_contains_required_schema_fields() -> None:
    """REQ-KONA-009: artifact builder emits every required Exp 1155 field."""
    artifact = build_hmc_compatibility_artifact(
        latent_dim=10,
        d1_error_mean=0.005,
        d2_variance=0.05,
        d3_disparity_ratio=2.0,
        d4_subspace_variance=0.02,
        d4_full_variance=0.03,
        gradient_method="numerical_fd",
    )

    required = {
        "latent_dim",
        "d1_symplectic_reversibility_error_mean",
        "d1_regime_signal",
        "d2_hamiltonian_variance",
        "d2_regime_signal",
        "d3_gradient_disparity_ratio",
        "d3_regime_signal",
        "d4_subspace_delta_h_variance",
        "d4_full_delta_h_variance",
        "d4_discrete_components_bottleneck",
        "gradient_method",
        "hmc_regime_classified",
        "hmc_regime",
        "recommended_sampler",
        "honest_verdict",
    }
    assert required.issubset(artifact)
    assert artifact["hmc_regime"] == "A"
    assert artifact["recommended_sampler"] == "hmc"
    assert artifact["honest_verdict"] == "regime_A_hmc_viable"
    json.dumps(artifact, allow_nan=False)


def test_build_artifact_uses_d4_for_regime_c_sampler_choice() -> None:
    """REQ-KONA-009: D4 bottleneck selects blocked Gibbs for Regime C."""
    artifact = build_hmc_compatibility_artifact(
        latent_dim=10,
        d1_error_mean=0.2,
        d2_variance=1.5,
        d3_disparity_ratio=150.0,
        d4_subspace_variance=0.01,
        d4_full_variance=1.5,
        gradient_method="numerical_fd",
    )

    assert artifact["hmc_regime"] == "C"
    assert artifact["d4_discrete_components_bottleneck"] is True
    assert artifact["recommended_sampler"] == "blocked_gibbs"
    assert artifact["honest_verdict"] == "regime_C_hmc_inappropriate"


def test_build_artifact_recommends_preconditioned_hmc_for_regime_b() -> None:
    """REQ-KONA-009: Regime B maps to the preconditioned HMC recommendation."""
    artifact = build_hmc_compatibility_artifact(
        latent_dim=10,
        d1_error_mean=0.005,
        d2_variance=0.2,
        d3_disparity_ratio=2.0,
        d4_subspace_variance=0.01,
        d4_full_variance=0.2,
        gradient_method="numerical_fd",
    )

    assert artifact["hmc_regime"] == "B"
    assert artifact["recommended_sampler"] == "preconditioned_hmc"
    assert artifact["honest_verdict"] == "regime_B_preconditioning_needed"


def test_gradient_disparity_ratio_handles_flat_and_zero_variance_components() -> None:
    """REQ-KONA-009: D3 handles flat components without non-finite ratios."""
    q_points = np.array([[0.0, 0.0], [0.5, -0.5], [1.0, 1.0]], dtype=np.float64)
    flat = LatentEnergyComponent("flat", lambda x: 1.0)
    ratio_flat, variances_flat = gradient_disparity_ratio([flat], q_points, fd_eps=1e-6)
    assert ratio_flat == pytest.approx(1.0)
    assert variances_flat["flat"] == pytest.approx(0.0)

    curved = _quadratic_component("curved")
    ratio_mixed, variances_mixed = gradient_disparity_ratio(
        [flat, curved],
        q_points,
        fd_eps=1e-6,
    )
    assert ratio_mixed >= 100.0
    assert variances_mixed["flat"] == pytest.approx(0.0)
    assert np.isfinite(ratio_mixed)


def test_default_component_builders_are_finite_and_named() -> None:
    """SCENARIO-KONA-008: default bridge exposes k=5 plus SemEnergy/ThinkPRM D4 names."""
    x = np.array([0.25, -0.5], dtype=np.float64)
    components = build_default_latent_components(latent_dim=2)
    subspace = build_default_continuous_subspace_components(latent_dim=2)

    assert [component.name for component in components] == [
        "SOSKANEnergyV3",
        "SemEnergyProbe",
        "ASTStructureVerifier",
        "SemanticConsistencyVerifier",
        "Z3MathVerifier",
    ]
    assert [component.name for component in subspace] == ["SemEnergyProbe", "ThinkPRMProbe"]
    assert all(np.isfinite(component.energy(x)) for component in components + subspace)
    assert build_default_latent_components(latent_dim=2)[0].name == "SOSKANEnergyV3"


def test_run_diagnostics_classifies_isotropic_quadratic_as_regime_a() -> None:
    """SCENARIO-KONA-008: smooth isotropic components produce Regime A signals."""
    components = [_quadratic_component(f"component_{idx}") for idx in range(5)]
    artifact = run_hmc_compatibility_diagnostics(
        latent_dim=2,
        components=components,
        continuous_components=components[:2],
        config=HMCCompatibilityConfig(
            n_diagnostic_points=12,
            seed=1155,
            d1_leapfrog_steps=4,
            d2_leapfrog_steps=4,
            step_size=0.01,
            fd_eps=1e-6,
        ),
    )

    assert artifact["latent_dim"] == 2
    assert artifact["gradient_method"] == "numerical_fd"
    assert artifact["d1_regime_signal"] == "A"
    assert artifact["d2_regime_signal"] == "A"
    assert artifact["d3_regime_signal"] == "A"
    assert artifact["hmc_regime"] == "A"


def test_load_latent_dim_from_exp1154_reads_required_field(tmp_path: Path) -> None:
    """REQ-KONA-009: Exp 1155 derives latent_dim from the Exp 1154 artifact."""
    path = tmp_path / "experiment_1154_snap_validity_sweep.json"
    path.write_text(json.dumps({"latent_dim": 10}), encoding="utf-8")
    assert load_latent_dim_from_exp1154(path) == 10


def test_script_main_writes_required_deliverable_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-KONA-008: experiment script writes the canonical JSON artifact."""
    exp1154_path = tmp_path / "experiment_1154_snap_validity_sweep.json"
    output_path = tmp_path / "experiment_1155_hmc_compatibility_diagnostics.json"
    exp1154_path.write_text(json.dumps({"latent_dim": 2}), encoding="utf-8")

    monkeypatch.setattr(experiment, "EXP1154_PATH", exp1154_path)
    monkeypatch.setattr(experiment, "RESULT_PATH", output_path)
    monkeypatch.setattr(
        experiment,
        "DEFAULT_CONFIG",
        HMCCompatibilityConfig(
            n_diagnostic_points=6,
            seed=1155,
            d1_leapfrog_steps=2,
            d2_leapfrog_steps=2,
            step_size=0.01,
            fd_eps=1e-6,
        ),
    )
    monkeypatch.setattr(
        experiment,
        "build_default_latent_components",
        lambda latent_dim: [_quadratic_component(f"component_{idx}") for idx in range(5)],
    )
    monkeypatch.setattr(
        experiment,
        "build_default_continuous_subspace_components",
        lambda latent_dim: [
            _quadratic_component("SemEnergyProbe"),
            _quadratic_component("ThinkPRMProbe"),
        ],
    )

    artifact = experiment.main()

    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["latent_dim"] == 2
    assert written["hmc_regime_classified"] is True
    assert written["gradient_method"] == "numerical_fd"
