"""Tests for Exp 1149 KV260 v5 DC-continuous Ising diagnostic.

Spec refs: REQ-HW-042, SCENARIO-HW-042.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import experiment_1149_kv260_v5_dc_continuous_diagnostic as exp1149


def test_dc_spectral_split_reconstructs_j_and_is_psd() -> None:
    """REQ-HW-042: positive/negative eigenspectrum split reconstructs J."""
    j_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])

    j_plus, j_minus = exp1149.dc_spectral_split(j_matrix)

    np.testing.assert_allclose(j_plus - j_minus, j_matrix, atol=1e-12)
    assert np.linalg.eigvalsh(j_plus).min() >= -1e-12
    assert np.linalg.eigvalsh(j_minus).min() >= -1e-12


def test_dc_update_matches_energy_gradient_and_clips() -> None:
    """REQ-HW-042: clipped DC update uses the split implied by E=-0.5*sJs."""
    j_matrix = np.array([[0.0, 2.0], [2.0, 0.0]])
    j_plus, j_minus = exp1149.dc_spectral_split(j_matrix)
    state = np.array([0.8, 0.8])

    updated = exp1149.dc_proximal_step(state, j_plus, j_minus, alpha=0.5)

    np.testing.assert_allclose(updated, np.array([1.0, 1.0]))


def test_threshold_zero_maps_to_plus_one() -> None:
    """REQ-HW-042: thresholding returns {-1,+1}, with zero tied to +1."""
    continuous = np.array([-0.2, 0.0, 3.0])

    spins = exp1149.threshold_spins(continuous)

    np.testing.assert_array_equal(spins, np.array([-1, 1, 1], dtype=np.int8))


def test_exp1134_seeded_j_matrices_are_three_sparse_symmetric_rings() -> None:
    """SCENARIO-HW-042: the diagnostic evaluates exactly three deterministic J matrices."""
    matrices = exp1149.build_exp1134_seeded_j_matrices()

    assert len(matrices) == 3
    for j_matrix in matrices:
        assert j_matrix.shape == (exp1149.N_SPINS, exp1149.N_SPINS)
        np.testing.assert_allclose(j_matrix, j_matrix.T)
        np.testing.assert_allclose(np.diag(j_matrix), 0.0)
        assert np.all(np.count_nonzero(j_matrix, axis=1) == exp1149.K_NEIGHBORS)


def test_sparse_tables_roundtrip_ring_matrix() -> None:
    """REQ-HW-042: sparse v4 tables preserve every non-zero ring coupling."""
    j_matrix = exp1149.build_exp1134_seeded_j_matrices()[0]

    nbr_idx, j_sparse = exp1149.sparse_tables_from_j(j_matrix, k_neighbors=exp1149.K_NEIGHBORS)
    reconstructed = exp1149.dense_from_sparse_tables(nbr_idx, j_sparse)

    np.testing.assert_allclose(reconstructed, j_matrix)


def test_ising_energy_and_constraint_accuracy() -> None:
    """REQ-HW-042: energy and final accuracy report the Ising constraints."""
    j_matrix = np.array(
        [
            [0.0, 1.0, -1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    satisfied = np.array([1, 1, -1], dtype=np.int8)
    one_wrong = np.array([1, -1, -1], dtype=np.int8)

    assert exp1149.ising_energy(j_matrix, satisfied) == pytest.approx(-2.0)
    assert exp1149.constraint_satisfaction_accuracy(j_matrix, satisfied) == pytest.approx(100.0)
    assert exp1149.constraint_satisfaction_accuracy(j_matrix, one_wrong) == pytest.approx(50.0)
    assert exp1149.constraint_satisfaction_accuracy(np.zeros((3, 3)), satisfied) == pytest.approx(
        100.0
    )


def test_kl_against_cpu_gibbs_is_zero_for_uniform_zero_energy() -> None:
    """REQ-HW-042: exact CPU Gibbs comparison is well-defined on all states."""
    j_matrix = np.zeros((3, 3), dtype=np.float64)
    samples = exp1149.all_spin_states(3)

    kl = exp1149.kl_against_cpu_gibbs(samples, j_matrix, beta=2.0)

    assert kl == pytest.approx(0.0)


def test_run_dc_sampler_smoke_reports_energy_time_accuracy() -> None:
    """SCENARIO-HW-042: one DC run emits thresholded samples and EDDP metrics."""
    j_matrix = exp1149.build_exp1134_seeded_j_matrices()[0]
    config = exp1149.DCConfig(n_restarts=32, max_iter=40, alpha=0.05, tolerance=1e-8, seed=99)

    measurement = exp1149.run_dc_measurement(j_matrix, matrix_id="toy", config=config)

    assert measurement["matrix_id"] == "toy"
    assert measurement["samples"].shape == (32, exp1149.N_SPINS)
    assert set(np.unique(measurement["samples"]).tolist()).issubset({-1, 1})
    assert np.isfinite(measurement["kl_v5_vs_cpu_gibbs"])
    assert np.isfinite(measurement["energy_at_convergence"])
    assert measurement["wall_clock_s"] >= 0.0
    assert 0.0 <= measurement["final_accuracy"] <= 100.0
    assert 0.0 <= measurement["convergence_fraction"] <= 1.0


def test_run_v5_alpha_grid_selects_best_alpha_and_covers_early_convergence() -> None:
    """REQ-HW-042: alpha grid returns summaries and handles immediate convergence."""
    j_matrix = exp1149.build_exp1134_seeded_j_matrices()[0]
    config = exp1149.DCConfig(n_restarts=8, max_iter=3, alpha=0.0, tolerance=1e-12, seed=5)

    best_alpha, measurements, summary = exp1149.run_v5_alpha_grid(
        [j_matrix],
        alpha_grid=(0.0, 0.01),
        base_config=config,
    )

    assert best_alpha in {0.0, 0.01}
    assert len(measurements) == 1
    assert len(summary) == 2
    assert {row["alpha"] for row in summary} == {0.0, 0.01}
    assert all("mean_final_accuracy" in row for row in summary)


def test_v4_sparse_measurement_smoke() -> None:
    """REQ-HW-042: v4 sparse comparison can be run on the same J matrix."""
    j_matrix = exp1149.build_exp1134_seeded_j_matrices()[0]

    measurement = exp1149.run_v4_sparse_measurement(j_matrix, n_record=32, burn_in=4, seed=7)

    assert measurement["kl_v4_sparse_vs_cpu_gibbs"] >= 0.0
    assert measurement["n_record"] == 32
    assert measurement["burn_in_sweeps"] == 4


def test_verdict_classifier_all_branches() -> None:
    """REQ-HW-042: honest verdict uses only the approved Exp 1149 labels."""
    assert exp1149.classify_verdict(0.01, diverged=False) == "kl_below_threshold_v5_viable"
    assert exp1149.classify_verdict(0.08, diverged=False) == "kl_improved_not_threshold"
    assert exp1149.classify_verdict(0.20, diverged=False) == "kl_unchanged_topology_wall"
    assert exp1149.classify_verdict(float("nan"), diverged=False) == "algorithm_diverged"
    assert exp1149.classify_verdict(0.01, diverged=True) == "algorithm_diverged"


def test_rtl_recommendation_tracks_verdict() -> None:
    """REQ-HW-042: RTL recommendation documents what v5 should implement next."""
    viable = exp1149.rtl_recommendation_for_verdict("kl_below_threshold_v5_viable")
    improved = exp1149.rtl_recommendation_for_verdict("kl_improved_not_threshold")
    unchanged = exp1149.rtl_recommendation_for_verdict("kl_unchanged_topology_wall")
    diverged = exp1149.rtl_recommendation_for_verdict("algorithm_diverged")

    assert "DC-continuous" in viable
    assert "stochastic" in improved
    assert "sequential Gibbs" in unchanged
    assert "Do not implement" in diverged


def test_build_artifact_has_required_schema() -> None:
    """SCENARIO-HW-042: artifact contains every field required by the roadmap."""
    measurements = [
        {
            "matrix_id": "j0",
            "alpha": 0.05,
            "kl_v5_vs_cpu_gibbs": 0.08,
            "energy_at_convergence": -1.0,
            "wall_clock_s": 0.01,
            "final_accuracy": 75.0,
            "convergence_fraction": 1.0,
            "n_restarts": 10,
            "iterations_mean": 4.0,
        },
        {
            "matrix_id": "j1",
            "alpha": 0.05,
            "kl_v5_vs_cpu_gibbs": 0.10,
            "energy_at_convergence": -2.0,
            "wall_clock_s": 0.02,
            "final_accuracy": 80.0,
            "convergence_fraction": 1.0,
            "n_restarts": 10,
            "iterations_mean": 5.0,
        },
    ]
    v4_measurements = [{"matrix_id": "j0", "kl_v4_sparse_vs_cpu_gibbs": 0.12}]

    artifact = exp1149.build_artifact(
        best_alpha=0.05,
        best_measurements=measurements,
        per_alpha_summary=[{"alpha": 0.05, "mean_kl_v5": 0.09}],
        v4_measurements=v4_measurements,
        duration_s=0.25,
        run_date="2026-05-02T00:00:00Z",
    )

    assert exp1149.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["algorithm"] == "dc_continuous_relaxation"
    assert artifact["kl_v4_best_prior"] == exp1149.KL_V4_BEST_PRIOR
    assert artifact["kl_v4_with_self_adaptive_prior"] == exp1149.KL_V4_WITH_SELF_ADAPTIVE_PRIOR
    assert artifact["kl_v5_best"] == pytest.approx(0.09)
    assert artifact["energy_time_accuracy_reported"] is True
    assert artifact["kv260_v5_diagnostic_complete"] is True


def test_prior_loader_reads_exp1134_artifact() -> None:
    """REQ-HW-042: prior constants are anchored to the existing exp1134 result."""
    prior = exp1149.load_v4_prior_artifact()

    assert prior["kl_v4_best"] == pytest.approx(0.1127718014422604)
    assert prior["kl_v4_with_self_adaptive"] == pytest.approx(31.893700066833517)


@pytest.mark.skipif(not exp1149.DELIVERABLE.exists(), reason="artifact not yet generated")
def test_deliverable_json_has_required_fields() -> None:
    """SCENARIO-HW-042: generated artifact satisfies the external result contract."""
    payload = json.loads(exp1149.DELIVERABLE.read_text())

    assert exp1149.REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["n_j_matrices_tested"] == 3
    assert payload["kv260_v5_diagnostic_complete"] is True
    assert payload["honest_verdict"] in exp1149.HONEST_VERDICTS
    assert Path(payload["v4_spec_path"]).as_posix() == "hardware/kv260/ising_sampler_v4_spec.md"
