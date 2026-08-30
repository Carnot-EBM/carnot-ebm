"""Tests for the matched single-site temporal exchange sampler.

Spec: REQ-SAMPLE-097, SCENARIO-SAMPLE-097, SCENARIO-SAMPLE-098.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from carnot.sampling import temporal_exchange as te


SPEC_PATH = Path("openspec/capabilities/training-inference/spec.md")


def _two_spin_problem() -> tuple[np.ndarray, np.ndarray]:
    biases = np.asarray([0.2, -0.1], dtype=np.float64)
    couplings = np.asarray([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)
    return biases, couplings


def test_req_sample_097_spec_precedes_temporal_sampler() -> None:
    """REQ-SAMPLE-097: The sampling capability owns the new behavior."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-SAMPLE-097" in text
    assert "SCENARIO-SAMPLE-097" in text
    assert "SCENARIO-SAMPLE-098" in text
    assert "SCENARIO-SAMPLE-099" in text


def test_scenario_sample_097_conditional_matches_exact_target_ratio() -> None:
    """SCENARIO-SAMPLE-097: The local logistic rule matches the Ising law."""

    biases, couplings = _two_spin_problem()
    current = np.asarray([-1, 1], dtype=np.int8)
    previous = np.asarray([1, -1], dtype=np.int8)
    temperature = 0.75

    observed = te.conditional_probability_plus(
        biases,
        couplings,
        current,
        previous,
        site=0,
        temperature=temperature,
        temporal_coupling=0.0,
    )
    field = biases[0] + couplings[0, 1] * current[1]
    assert observed == pytest.approx(1.0 / (1.0 + math.exp(-2.0 * field / temperature)))

    states, probabilities, _energies = te.enumerate_target_distribution(
        biases, couplings, temperature
    )
    mask = states[:, 1] == 1
    exact = probabilities[mask & (states[:, 0] == 1)].sum() / probabilities[mask].sum()
    assert observed == pytest.approx(exact)


def test_scenario_sample_097_previous_state_is_explicit_and_sweep_frozen() -> None:
    """SCENARIO-SAMPLE-097: Prior spins change only after a completed sweep."""

    biases, couplings = _two_spin_problem()
    state = te.initialize_temporal_state(current=[-1, 1], previous=[1, -1])
    generator = np.random.default_rng(17)
    original_previous = state.previous.copy()

    te.attempt_single_site_update(
        state,
        biases,
        couplings,
        temperature=1.0,
        temporal_coupling=-0.08,
        generator=generator,
    )
    np.testing.assert_array_equal(state.previous, original_previous)
    assert state.sweep_position == 1
    assert state.update_count == 1

    te.attempt_single_site_update(
        state,
        biases,
        couplings,
        temperature=1.0,
        temporal_coupling=-0.08,
        generator=generator,
    )
    np.testing.assert_array_equal(state.previous, state.current)
    assert state.sweep_position == 0
    assert state.update_count == 2


def test_scenario_sample_097_coupling_sign_controls_flip_or_persistence() -> None:
    """SCENARIO-SAMPLE-097: AFM and FM signs shift the conditional correctly."""

    biases = np.zeros(1, dtype=np.float64)
    couplings = np.zeros((1, 1), dtype=np.float64)
    current = np.asarray([-1], dtype=np.int8)
    previous = np.asarray([1], dtype=np.int8)
    antiferromagnetic = te.conditional_probability_plus(
        biases,
        couplings,
        current,
        previous,
        site=0,
        temperature=0.75,
        temporal_coupling=-0.08,
    )
    uncoupled = te.conditional_probability_plus(
        biases,
        couplings,
        current,
        previous,
        site=0,
        temperature=0.75,
        temporal_coupling=0.0,
    )
    ferromagnetic = te.conditional_probability_plus(
        biases,
        couplings,
        current,
        previous,
        site=0,
        temperature=0.75,
        temporal_coupling=0.08,
    )
    assert antiferromagnetic < uncoupled < ferromagnetic


def test_scenario_sample_098_matched_updates_seeds_and_zero_coupling_equivalence() -> None:
    """SCENARIO-SAMPLE-098: Disabled temporal coupling is ordinary Gibbs."""

    biases, couplings = _two_spin_problem()
    common = {
        "biases": biases,
        "couplings": couplings,
        "current": [-1, 1],
        "previous": [1, -1],
        "temperature": 0.75,
        "seed": 679300,
        "burn_in_sweeps": 3,
        "n_samples": 12,
        "sweeps_per_sample": 2,
        "optimum_energy": -0.5,
    }
    ordinary = te.sample_ising(arm="ordinary_gibbs", temporal_coupling=0.0, **common)
    disabled = te.sample_ising(
        arm="temporal_exchange_zero_coupling", temporal_coupling=0.0, **common
    )
    repeat = te.sample_ising(arm="ordinary_gibbs", temporal_coupling=0.0, **common)

    expected_updates = 2 * (3 + 12 * 2)
    assert ordinary.update_count == disabled.update_count == expected_updates
    np.testing.assert_array_equal(
        ordinary.collection_update_counts,
        np.arange(3 * 2 + 2 * 2, expected_updates + 1, 2 * 2),
    )
    np.testing.assert_array_equal(ordinary.samples, disabled.samples)
    np.testing.assert_array_equal(ordinary.energy_trace, disabled.energy_trace)
    np.testing.assert_array_equal(ordinary.samples, repeat.samples)
    assert ordinary.trajectory_sha256 == disabled.trajectory_sha256
    assert ordinary.optimum_hitting_updates == disabled.optimum_hitting_updates


def test_req_sample_097_exact_enumerator_normalizes_and_rejects_large_state_space() -> None:
    """REQ-SAMPLE-097: Headline target laws are exhaustive and bounded."""

    biases, couplings = _two_spin_problem()
    states, probabilities, energies = te.enumerate_target_distribution(
        biases, couplings, temperature=2.0
    )
    assert states.shape == (4, 2)
    assert energies.shape == (4,)
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.all(probabilities > 0.0)
    np.testing.assert_allclose(
        energies,
        np.asarray([te.ising_energy(state, biases, couplings) for state in states]),
    )

    with pytest.raises(te.TemporalExchangeInputError, match="exceeds"):
        te.enumerate_target_distribution(
            np.zeros(13), np.zeros((13, 13)), temperature=1.0, maximum_states=4096
        )


def test_req_sample_097_rejects_ambiguous_sampler_inputs() -> None:
    """REQ-SAMPLE-097: Invalid state, graph, schedule, and arm inputs fail closed."""

    biases, couplings = _two_spin_problem()
    with pytest.raises(te.TemporalExchangeInputError, match="symmetric"):
        te.conditional_probability_plus(
            biases,
            np.asarray([[0.0, 0.2], [0.4, 0.0]]),
            [-1, 1],
            [1, -1],
            site=0,
            temperature=1.0,
            temporal_coupling=0.0,
        )
    with pytest.raises(te.TemporalExchangeInputError, match="bipolar"):
        te.initialize_temporal_state(current=[0, 1], previous=[1, -1])
    with pytest.raises(te.TemporalExchangeInputError, match="temperature"):
        te.enumerate_target_distribution(biases, couplings, temperature=0.0)
    with pytest.raises(te.TemporalExchangeInputError, match="unknown arm"):
        te.sample_ising(
            biases,
            couplings,
            current=[-1, 1],
            previous=[1, -1],
            temperature=1.0,
            arm="unknown",
            temporal_coupling=0.0,
            seed=1,
            burn_in_sweeps=1,
            n_samples=1,
        )
    with pytest.raises(te.TemporalExchangeInputError, match="zero coupling"):
        te.sample_ising(
            biases,
            couplings,
            current=[-1, 1],
            previous=[1, -1],
            temperature=1.0,
            arm="ordinary_gibbs",
            temporal_coupling=0.08,
            seed=1,
            burn_in_sweeps=1,
            n_samples=1,
        )


def test_req_sample_097_all_input_guards_have_observable_failures() -> None:
    """REQ-SAMPLE-097: Each public graph and state guard reports its cause."""

    biases, couplings = _two_spin_problem()
    conditional_common = {
        "biases": biases,
        "couplings": couplings,
        "current": [-1, 1],
        "previous": [1, -1],
        "temperature": 1.0,
        "temporal_coupling": 0.0,
    }
    with pytest.raises(te.TemporalExchangeInputError, match="one-dimensional"):
        te.initialize_temporal_state(current=[[-1, 1]], previous=[1, -1])
    with pytest.raises(te.TemporalExchangeInputError, match="biases"):
        te.conditional_probability_plus(
            [float("nan"), 0.0],
            couplings,
            [-1, 1],
            [1, -1],
            site=0,
            temperature=1.0,
            temporal_coupling=0.0,
        )
    with pytest.raises(te.TemporalExchangeInputError, match="square graph"):
        te.conditional_probability_plus(
            biases,
            np.zeros((2, 3)),
            [-1, 1],
            [1, -1],
            site=0,
            temperature=1.0,
            temporal_coupling=0.0,
        )
    with pytest.raises(te.TemporalExchangeInputError, match="diagonal"):
        te.conditional_probability_plus(
            biases,
            np.eye(2),
            [-1, 1],
            [1, -1],
            site=0,
            temperature=1.0,
            temporal_coupling=0.0,
        )
    with pytest.raises(te.TemporalExchangeInputError, match="site"):
        te.conditional_probability_plus(site=2, **conditional_common)
    with pytest.raises(te.TemporalExchangeInputError, match="coupling must be finite"):
        te.conditional_probability_plus(
            site=0, **{**conditional_common, "temporal_coupling": float("nan")}
        )
    with pytest.raises(te.TemporalExchangeInputError, match="does not match"):
        bad_state = te.initialize_temporal_state([-1], [1])
        te.attempt_single_site_update(
            bad_state,
            biases,
            couplings,
            temperature=1.0,
            temporal_coupling=0.0,
            generator=np.random.default_rng(1),
        )
    with pytest.raises(te.TemporalExchangeInputError, match="match the bias"):
        te.ising_energy([-1, 1], biases, np.zeros((1, 1)))

    sample_common = {
        "biases": biases,
        "couplings": couplings,
        "current": [-1, 1],
        "previous": [1, -1],
        "temperature": 1.0,
        "arm": "temporal_exchange",
        "seed": 1,
        "burn_in_sweeps": 1,
        "n_samples": 1,
    }
    with pytest.raises(te.TemporalExchangeInputError, match="coupling must be finite"):
        te.sample_ising(temporal_coupling=float("nan"), **sample_common)
    with pytest.raises(te.TemporalExchangeInputError, match="collection counts"):
        te.sample_ising(temporal_coupling=0.0, **{**sample_common, "n_samples": 0})
    with pytest.raises(te.TemporalExchangeInputError, match="initial state pair"):
        te.sample_ising(
            temporal_coupling=0.0,
            **{**sample_common, "current": [-1], "previous": [1]},
        )
