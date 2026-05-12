"""Tests for Exp 1961 mixed-variable IGD sampling.

Spec traces: REQ-IGD-1961, REQ-IGD-1961-1, REQ-IGD-1961-2,
REQ-IGD-1961-3, REQ-IGD-1961-4, SCENARIO-IGD-1961.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from carnot.samplers.interleaved_gibbs_diffusion import (
    BenchmarkConfig,
    InterleavedGibbsDiffusionSampler,
    Max3SatInstance,
    SequentialGibbsSampler,
    generate_synthetic_max3sat,
    potts_one_hot,
    run_max3sat_benchmark,
)


def test_req_igd_1961_spec_entry_exists() -> None:
    spec = Path("openspec/capabilities/igd/spec.md").read_text()
    assert "REQ-IGD-1961" in spec
    assert "SCENARIO-IGD-1961" in spec


def test_req_igd_1961_synthetic_max3sat_uses_potts_encoding() -> None:
    """REQ-IGD-1961-3: deterministic 3-literal clauses and q=2 Potts states."""
    problem = generate_synthetic_max3sat(num_variables=8, num_clauses=20, seed=1961)

    assert isinstance(problem, Max3SatInstance)
    assert problem.q == 2
    assert problem.clauses.shape == (20, 3)
    assert np.all(problem.clauses != 0)
    assert np.max(np.abs(problem.clauses)) <= problem.num_variables
    assert problem.count_satisfied(problem.planted_assignment) == problem.num_clauses

    encoded = potts_one_hot(problem.planted_assignment, q=2)
    assert encoded.shape == (8, 2)
    assert np.all(encoded.sum(axis=1) == 1.0)


def test_req_igd_1961_interleaves_noise_and_discrete_updates() -> None:
    """REQ-IGD-1961-1/2: IGD maintains logits and injects finite noise per sweep."""
    problem = generate_synthetic_max3sat(num_variables=10, num_clauses=30, seed=7)
    init_state = np.zeros(problem.num_variables, dtype=np.int64)
    sampler = InterleavedGibbsDiffusionSampler(
        beta=3.0,
        noise_scale=0.2,
        logit_coupling=0.6,
        seed=11,
    )

    run = sampler.sample(problem, n_sweeps=12, init_state=init_state)

    assert run.sampler_name == "interleaved_gibbs_diffusion"
    assert run.final_state.shape == (problem.num_variables,)
    assert run.final_logits.shape == (problem.num_variables, problem.q)
    assert len(run.satisfaction_history) == 13
    assert len(run.continuous_noise_norms) == 12
    assert all(np.isfinite(run.continuous_noise_norms))
    assert any(norm > 0.0 for norm in run.continuous_noise_norms)
    assert np.all(np.isfinite(run.final_logits))
    assert set(np.unique(run.final_state)).issubset({0, 1})
    assert run.best_satisfied >= run.satisfaction_history[0]


def test_req_igd_1961_sequential_gibbs_baseline_reports_matching_metrics() -> None:
    """REQ-IGD-1961-4: baseline returns the same mixing/convergence fields."""
    problem = generate_synthetic_max3sat(num_variables=9, num_clauses=27, seed=9)
    baseline = SequentialGibbsSampler(beta=3.0, seed=12)

    run = baseline.sample(problem, n_sweeps=10, init_state=np.zeros(problem.num_variables, dtype=np.int64))

    assert run.sampler_name == "sequential_gibbs"
    assert run.final_state.shape == (problem.num_variables,)
    assert run.final_logits.shape == (problem.num_variables, problem.q)
    assert len(run.satisfaction_history) == 11
    assert len(run.continuous_noise_norms) == 0
    assert run.mixing_time is None or 0 <= run.mixing_time <= 10
    assert np.isfinite(run.convergence_rate)
    assert 0 <= run.best_satisfied <= problem.num_clauses


def test_req_igd_1961_sampler_can_initialize_state_from_seed() -> None:
    """REQ-IGD-1961-1: omitted init_state still produces valid Potts states."""
    problem = generate_synthetic_max3sat(num_variables=7, num_clauses=18, seed=21)
    sampler = InterleavedGibbsDiffusionSampler(seed=22)

    run = sampler.sample(problem, n_sweeps=3)

    assert run.final_state.shape == (problem.num_variables,)
    assert set(np.unique(run.final_state)).issubset({0, 1})


def test_scenario_igd_1961_benchmark_artifact_payload_is_deterministic() -> None:
    """SCENARIO-IGD-1961: benchmark records IGD and sequential Gibbs metrics."""
    config = BenchmarkConfig(
        num_variables=12,
        num_clauses=42,
        n_sweeps=28,
        seed=1961,
        beta=3.5,
        noise_scale=0.15,
        logit_coupling=0.7,
        target_satisfaction_ratio=0.95,
    )

    first = run_max3sat_benchmark(config)
    second = run_max3sat_benchmark(config)

    assert first == second
    assert first["experiment_id"] == "1961"
    assert first["spec_refs"] == [
        "REQ-IGD-1961",
        "REQ-IGD-1961-1",
        "REQ-IGD-1961-2",
        "REQ-IGD-1961-3",
        "REQ-IGD-1961-4",
        "REQ-IGD-1961-5",
        "SCENARIO-IGD-1961",
    ]
    assert first["problem"]["q"] == 2
    assert first["problem"]["clause_width"] == 3
    assert set(first["metrics"]) == {"igd", "sequential_gibbs"}
    assert first["metrics"]["igd"]["finite_logits"] is True
    assert first["metrics"]["igd"]["continuous_noise_injections"] == config.n_sweeps
    assert first["metrics"]["sequential_gibbs"]["continuous_noise_injections"] == 0
    assert first["comparison"]["mixing_time_delta_sweeps"] is None or isinstance(
        first["comparison"]["mixing_time_delta_sweeps"], int
    )
    assert first["honest_verdict"] in {
        "igd_mixed_sampler_benchmark_complete",
        "igd_mixed_sampler_no_baseline_improvement",
    }

    no_mix = run_max3sat_benchmark(
        BenchmarkConfig(
            num_variables=8,
            num_clauses=18,
            n_sweeps=4,
            seed=23,
            target_satisfaction_ratio=1.01,
        )
    )
    assert no_mix["comparison"]["mixing_time_delta_sweeps"] is None
