"""Tests for Exp 1161 KV260 v6 sequential Gibbs correctness pivot.

Spec refs: REQ-HW-045, SCENARIO-HW-045.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import experiment_1149_kv260_v5_dc_continuous_diagnostic as exp1149
from scripts import experiment_1161_kv260_v6_sequential_gibbs as exp1161


def test_sequential_sampler_matches_cpu_reference_and_round_robin_order() -> None:
    """REQ-HW-045: v6 updates exactly one spin per step in t mod N order."""
    j_matrix = np.array(
        [
            [0.0, 0.4, -0.2],
            [0.4, 0.0, 0.6],
            [-0.2, 0.6, 0.0],
        ],
        dtype=np.float64,
    )
    b = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    sampler = exp1161.SequentialGibbsSampler()

    samples = sampler.sample(j_matrix, b, n_spins=3, n_steps=9, beta=1.7, seed=123)
    reference = exp1161.cpu_gibbs_reference_samples(
        j_matrix, b, n_spins=3, n_steps=9, beta=1.7, seed=123
    )

    np.testing.assert_array_equal(samples, reference)
    np.testing.assert_array_equal(sampler.last_update_indices, np.arange(9) % 3)
    assert samples.shape == (9, 3)
    assert samples.dtype == np.int8
    assert set(np.unique(samples).tolist()).issubset({-1, 1})


def test_sampler_rejects_malformed_inputs() -> None:
    """REQ-HW-045: sampler refuses J/b shapes that cannot define an Ising model."""
    sampler = exp1161.SequentialGibbsSampler()
    with pytest.raises(ValueError, match="J"):
        sampler.sample(np.zeros((2, 3)), np.zeros(2), n_spins=2, n_steps=4, beta=1.0, seed=0)
    with pytest.raises(ValueError, match="b"):
        sampler.sample(np.zeros((2, 2)), np.zeros(3), n_spins=2, n_steps=4, beta=1.0, seed=0)
    with pytest.raises(ValueError, match="n_steps"):
        sampler.sample(np.zeros((2, 2)), np.zeros(2), n_spins=2, n_steps=-1, beta=1.0, seed=0)


def test_exp1149_seeded_matrices_are_reused_exactly() -> None:
    """SCENARIO-HW-045: v6 uses the same three N=8 J matrices as Exp 1149."""
    matrices = exp1161.build_n8_exp1149_j_matrices()
    prior_matrices = exp1149.build_exp1134_seeded_j_matrices()

    assert len(matrices) == 3
    for actual, expected in zip(matrices, prior_matrices, strict=True):
        np.testing.assert_allclose(actual, expected)
        assert actual.shape == (8, 8)
        np.testing.assert_allclose(actual, actual.T)
        np.testing.assert_allclose(np.diag(actual), 0.0)
        assert np.all(np.count_nonzero(actual, axis=1) == 2)


def test_n128_k16_sparse_graph_is_symmetric_ring_topology() -> None:
    """REQ-HW-045: N=128 K=16 topology matches the sparse KV260 v4 target shape."""
    j_matrix = exp1161.build_sparse_ring_j_matrix(n_spins=128, k_neighbors=16, j_value=-1.0)

    assert j_matrix.shape == (128, 128)
    np.testing.assert_allclose(j_matrix, j_matrix.T)
    np.testing.assert_allclose(np.diag(j_matrix), 0.0)
    assert np.all(np.count_nonzero(j_matrix, axis=1) == 16)
    with pytest.raises(ValueError, match="even"):
        exp1161.build_sparse_ring_j_matrix(n_spins=8, k_neighbors=3)
    with pytest.raises(ValueError, match="less than"):
        exp1161.build_sparse_ring_j_matrix(n_spins=8, k_neighbors=8)


def test_empirical_kl_between_matching_sample_sets_is_zero() -> None:
    """REQ-HW-045: v6-vs-CPU KL is zero when both algorithms produce the same samples."""
    empty = np.empty((0, 3), dtype=np.int8)
    samples = np.array(
        [
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, -1],
        ],
        dtype=np.int8,
    )
    shifted = samples.copy()
    shifted[0, 0] = -1

    assert exp1161.empirical_kl_between_samples(empty, empty) == pytest.approx(0.0)
    assert exp1161.empirical_kl_between_samples(samples, samples) == pytest.approx(0.0)
    assert exp1161.empirical_kl_between_samples(samples, shifted) > 0.0


def test_n8_measurement_rejects_failed_matrix_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-HW-045: matrix-generation failure maps to a hard diagnostic error."""
    monkeypatch.setattr(exp1161, "build_n8_exp1149_j_matrices", lambda: [])

    with pytest.raises(ValueError, match="expected three"):
        exp1161.run_n8_measurements(n_steps=8, beta=2.0)


def test_short_n8_measurement_is_below_threshold() -> None:
    """SCENARIO-HW-045: the three N=8 v6-vs-CPU KL values satisfy the gate."""
    measurements = exp1161.run_n8_measurements(n_steps=96, beta=2.0)

    assert len(measurements) == 3
    assert exp1161.mean_kl(measurements, "kl_v6_vs_cpu_gibbs") == pytest.approx(0.0)
    assert all(m["kl_v6_below_threshold"] is True for m in measurements)
    assert {m["matrix_seed"] for m in measurements} == {1134, 1135, 1136}


def test_short_n128_measurement_is_below_threshold() -> None:
    """REQ-HW-045: the N=128 K=16 v6-vs-CPU sparse-topology KL satisfies the gate."""
    measurement = exp1161.run_n128_k16_measurement(n_steps=96, beta=2.0)

    assert measurement["n128_k16_tested"] is True
    assert measurement["n_spins"] == 128
    assert measurement["k_neighbors"] == 16
    assert measurement["kl_v6_vs_cpu_gibbs"] == pytest.approx(0.0)
    assert measurement["kl_v6_below_threshold"] is True


def test_rtl_spec_writer_emits_one_spin_per_clock_pseudocode(tmp_path: Path) -> None:
    """SCENARIO-HW-045: RTL pseudocode documents s[N], h[N], and one spin per cycle."""
    out = tmp_path / "ising_sampler_v6_spec.md"

    written = exp1161.write_rtl_spec(out)
    text = out.read_text()

    assert written is True
    assert "REQ-HW-045" in text
    assert "one spin per clock" in text
    assert "s[N]" in text
    assert "h[N]" in text
    assert "t % N" in text


def test_verdict_classifier_all_branches() -> None:
    """REQ-HW-045: honest verdict uses only the approved Exp 1161 vocabulary."""
    assert exp1161.classify_verdict(0.0) == "kl_near_zero_algorithm_correct"
    assert exp1161.classify_verdict(0.01) == "kl_below_threshold_sequential_correct"
    assert exp1161.classify_verdict(0.10) == "kl_above_threshold_unexplained"
    assert exp1161.classify_verdict(float("nan")) == "kl_above_threshold_unexplained"
    assert (
        exp1161.classify_verdict(0.0, matrix_generation_failed=True) == "matrix_generation_failed"
    )


def test_build_artifact_has_required_schema() -> None:
    """SCENARIO-HW-045: artifact contains every field required by the roadmap."""
    n8_measurements = [
        {
            "matrix_id": "exp1134_seeded_j0",
            "matrix_seed": 1134,
            "n_spins": 8,
            "n_steps": 10,
            "kl_v6_vs_cpu_gibbs": 0.0,
            "kl_v6_below_threshold": True,
        },
        {
            "matrix_id": "exp1134_seeded_j1",
            "matrix_seed": 1135,
            "n_spins": 8,
            "n_steps": 10,
            "kl_v6_vs_cpu_gibbs": 0.0,
            "kl_v6_below_threshold": True,
        },
        {
            "matrix_id": "exp1134_seeded_j2",
            "matrix_seed": 1136,
            "n_spins": 8,
            "n_steps": 10,
            "kl_v6_vs_cpu_gibbs": 0.0,
            "kl_v6_below_threshold": True,
        },
    ]
    n128_measurement = {
        "n128_k16_tested": True,
        "n_spins": 128,
        "k_neighbors": 16,
        "n_steps": 10,
        "kl_v6_vs_cpu_gibbs": 0.0,
        "kl_v6_below_threshold": True,
    }

    artifact = exp1161.build_artifact(
        n8_measurements=n8_measurements,
        n128_measurement=n128_measurement,
        kl_v5_prior=0.447,
        kl_v4_prior=0.1128,
        rtl_spec_written=True,
        duration_s=0.25,
        run_date="2026-05-02T00:00:00Z",
    )

    assert exp1161.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["algorithm"] == "sequential_gibbs"
    assert artifact["n_j_matrices_n8"] == 3
    assert artifact["kl_v6_vs_cpu_n8_mean"] == pytest.approx(0.0)
    assert artifact["kl_v6_below_threshold_n8"] is True
    assert artifact["n128_k16_tested"] is True
    assert artifact["kl_v6_vs_cpu_n128_mean"] == pytest.approx(0.0)
    assert artifact["kl_v6_below_threshold_n128"] is True
    assert artifact["kl_improvement_over_v5"] == pytest.approx(0.447)
    assert artifact["kl_improvement_over_v4"] == pytest.approx(0.1128)
    assert artifact["rtl_spec_path"] == "hardware/kv260/ising_sampler_v6_spec.md"
    assert artifact["honest_verdict"] == "kl_near_zero_algorithm_correct"


def test_main_writes_artifact_and_rtl_spec_to_configured_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-HW-045: main writes the deliverable JSON and v6 RTL spec."""
    artifact_path = tmp_path / "experiment_1161.json"
    rtl_path = tmp_path / "ising_sampler_v6_spec.md"
    monkeypatch.setattr(exp1161, "DELIVERABLE", artifact_path)
    monkeypatch.setattr(exp1161, "RTL_SPEC_PATH", rtl_path)
    monkeypatch.setattr(exp1161, "DEFAULT_N_STEPS", 32)

    assert exp1161.main() == 0

    payload = json.loads(artifact_path.read_text())
    assert exp1161.REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["algorithm"] == "sequential_gibbs"
    assert payload["kl_v6_below_threshold_n8"] is True
    assert payload["rtl_spec_written"] is True
    assert rtl_path.exists()


@pytest.mark.skipif(not exp1161.DELIVERABLE.exists(), reason="artifact not yet generated")
def test_deliverable_json_has_required_fields() -> None:
    """SCENARIO-HW-045: generated artifact satisfies the external result contract."""
    payload = json.loads(exp1161.DELIVERABLE.read_text())

    assert exp1161.REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["algorithm"] == "sequential_gibbs"
    assert payload["n_j_matrices_n8"] == 3
    assert payload["kl_v6_below_threshold_n8"] is True
    assert payload["n128_k16_tested"] is True
    assert payload["kl_v6_below_threshold_n128"] is True
    assert payload["kv260_v6_kl_below_threshold_sequential_gibbs"] is True
    assert payload["honest_verdict"] in exp1161.HONEST_VERDICTS
