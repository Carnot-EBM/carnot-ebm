"""Tests for the Z1 analog beta-drift detailed-balance correction prototype.

Spec coverage: REQ-SAMPLE-063, SCENARIO-SAMPLE-091.
"""

from __future__ import annotations

import json

import numpy as np

from carnot.samplers.backend import SamplerBackend
from carnot.sampling.z1_drift_correction import (
    DriftCorrectionConfig,
    SyntheticDriftIsingBackend,
    build_bipartite_ring_problem,
    build_exp1583_payload,
    combined_sigma,
    empirical_kl_proxy,
    empirical_ks_proxy,
    energy_bias,
    hamiltonian_energy,
    hamiltonian_score,
    hastings_log_acceptance,
    make_beta_drift,
    magnetization,
    magnetization_bias,
    proposal_log_probability,
    write_exp1583_artifact,
)


def test_req_sample_063_beta_drift_and_n128_problem_shape() -> None:
    """REQ-SAMPLE-063: simulator builds n=128 with controlled +/-5% beta drift."""

    config = DriftCorrectionConfig(n_samples=4, n_warmup_sweeps=2, sweeps_per_sample=1)
    biases, couplings = build_bipartite_ring_problem(config)
    drift = make_beta_drift(config)

    assert biases.shape == (128,)
    assert couplings.shape == (128, 128)
    assert np.allclose(couplings, couplings.T)
    assert np.allclose(np.diag(couplings), 0.0)
    assert drift.shape == (128,)
    assert np.isclose(drift.mean(), 1.0)
    assert np.isclose(drift.std(ddof=0), 0.05)


def test_scenario_sample_091_backend_boundary_and_sampling_trace() -> None:
    """SCENARIO-SAMPLE-091: drift simulator and correction expose a SamplerBackend boundary."""

    config = DriftCorrectionConfig(
        n_spins=16,
        n_samples=6,
        n_warmup_sweeps=4,
        sweeps_per_sample=2,
        drift_std=0.10,
    )
    biases, couplings = build_bipartite_ring_problem(config)
    drift = make_beta_drift(config)
    backend = SyntheticDriftIsingBackend(config=config, beta_multipliers=drift, corrected=True)

    samples = backend.sample(biases, couplings, n_samples=6, config={"beta": config.beta})
    minimized = backend.minimize_energy(biases, couplings, n_samples=3, n_steps=4, beta=config.beta)

    assert isinstance(backend, SamplerBackend)
    assert backend.backend_name == "synthetic-drift-ising-hastings"
    assert samples.shape == (6, 16)
    assert samples.dtype == bool
    assert minimized.shape == (3, 16)
    assert minimized.dtype == bool
    assert 0.0 <= backend.last_acceptance_rate <= 1.0


def test_req_sample_063_hastings_ratio_satisfies_detailed_balance() -> None:
    """REQ-SAMPLE-063: Hastings correction balances drifted proposals against target beta."""

    config = DriftCorrectionConfig(n_spins=8, beta=0.9, drift_std=0.25)
    biases, couplings = build_bipartite_ring_problem(config)
    drift = make_beta_drift(config)
    source = np.array([True, False, True, False, False, True, False, True])
    proposed = source.copy()
    block = np.array([0, 2, 4, 6])
    proposed[block] = ~proposed[block]

    forward_log_q = proposal_log_probability(source, proposed, block, biases, couplings, config.beta, drift)
    reverse_log_q = proposal_log_probability(proposed, source, block, biases, couplings, config.beta, drift)
    forward_log_a = min(
        0.0,
        hastings_log_acceptance(source, proposed, block, biases, couplings, config.beta, drift),
    )
    reverse_log_a = min(
        0.0,
        hastings_log_acceptance(proposed, source, block, biases, couplings, config.beta, drift),
    )
    source_log_pi = config.beta * hamiltonian_score(source, biases, couplings)
    proposed_log_pi = config.beta * hamiltonian_score(proposed, biases, couplings)

    assert np.isclose(
        source_log_pi + forward_log_q + forward_log_a,
        proposed_log_pi + reverse_log_q + reverse_log_a,
    )


def test_req_sample_063_exact_target_proposal_is_always_accepted() -> None:
    """REQ-SAMPLE-063: no-drift block-Gibbs proposal has Hastings acceptance one."""

    config = DriftCorrectionConfig(n_spins=8, beta=0.7, drift_std=0.0)
    biases, couplings = build_bipartite_ring_problem(config)
    drift = make_beta_drift(config)
    source = np.array([True, False, True, False, False, True, False, True])
    proposed = source.copy()
    block = np.array([1, 3, 5, 7])
    proposed[block] = ~proposed[block]

    assert hastings_log_acceptance(source, proposed, block, biases, couplings, config.beta, drift) == 0.0


def test_scenario_sample_091_payload_and_artifact_schema(tmp_path) -> None:
    """SCENARIO-SAMPLE-091: Exp 1583 payload carries required simulator-only fields."""

    config = DriftCorrectionConfig(
        n_spins=16,
        n_samples=64,
        n_warmup_sweeps=32,
        sweeps_per_sample=2,
        drift_std=0.15,
    )
    payload = build_exp1583_payload(config)
    artifact_path = tmp_path / "experiment_1583.json"
    write_exp1583_artifact(artifact_path, config)
    written = json.loads(artifact_path.read_text(encoding="utf-8"))

    required_fields = {
        "status",
        "synthetic_drift_simulator_ready",
        "correction_method",
        "uncorrected_energy_bias",
        "corrected_energy_bias",
        "uncorrected_magnetization_bias",
        "corrected_magnetization_bias",
        "correction_within_1sigma",
        "detailed_balance_correction_ready",
        "simulator_only_no_hardware_claim",
        "honest_verdict",
    }
    assert required_fields.issubset(payload)
    assert written == payload
    assert payload["status"] == "complete"
    assert payload["synthetic_drift_simulator_ready"] is True
    assert payload["correction_method"] == "hastings_boundary_accept_reject"
    assert payload["simulator_only_no_hardware_claim"] is True


def test_req_sample_063_metric_helpers() -> None:
    """REQ-SAMPLE-063: metric helpers report energy, magnetization, KL, KS, and sigma."""

    samples = np.array(
        [
            [True, False, True, False],
            [True, True, False, False],
            [False, False, True, True],
        ]
    )
    reference = np.array(
        [
            [True, False, False, False],
            [True, False, True, True],
            [False, True, False, True],
        ]
    )
    biases = np.array([0.1, -0.2, 0.3, -0.1])
    couplings = np.zeros((4, 4), dtype=float)
    couplings[0, 2] = couplings[2, 0] = 0.25

    sample_energy = hamiltonian_energy(samples, biases, couplings)
    reference_energy = hamiltonian_energy(reference, biases, couplings)

    assert sample_energy.shape == (3,)
    assert magnetization(samples).shape == (3,)
    assert energy_bias(sample_energy, reference_energy) == float(sample_energy.mean() - reference_energy.mean())
    assert magnetization_bias(samples, reference) == float(magnetization(samples).mean() - magnetization(reference).mean())
    assert empirical_kl_proxy(sample_energy, reference_energy) >= 0.0
    assert empirical_ks_proxy(sample_energy, reference_energy) >= 0.0
    assert combined_sigma(sample_energy, reference_energy) > 0.0
