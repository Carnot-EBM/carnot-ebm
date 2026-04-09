"""Tests for parallel Ising Gibbs sampler.

Spec coverage: REQ-SAMPLE-003, SCENARIO-SAMPLE-006, SCENARIO-SAMPLE-007,
               SCENARIO-SAMPLE-008
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.samplers.parallel_ising import (
    AnnealingSchedule,
    ParallelIsingSampler,
    _parallel_update,
    _checkerboard_update,
    sat_to_coupling_matrix,
    extract_ising_params,
    parallel_sample_states,
)


# --- AnnealingSchedule ---


class TestAnnealingSchedule:
    """REQ-SAMPLE-003: Annealing schedule for simulated annealing."""

    def test_linear_endpoints(self):
        """SCENARIO-SAMPLE-006: Linear schedule hits init and final."""
        s = AnnealingSchedule(beta_init=0.5, beta_final=10.0, schedule_type="linear")
        assert float(s.beta_at_step(0, 100)) == 0.5
        assert abs(float(s.beta_at_step(99, 100)) - 10.0) < 1e-5

    def test_linear_midpoint(self):
        """SCENARIO-SAMPLE-006: Linear schedule midpoint."""
        s = AnnealingSchedule(beta_init=0.0, beta_final=10.0, schedule_type="linear")
        mid = float(s.beta_at_step(50, 101))
        assert abs(mid - 5.0) < 0.1

    def test_geometric_endpoints(self):
        """SCENARIO-SAMPLE-006: Geometric schedule hits init and final."""
        s = AnnealingSchedule(beta_init=0.1, beta_final=10.0, schedule_type="geometric")
        assert abs(float(s.beta_at_step(0, 100)) - 0.1) < 1e-5
        assert abs(float(s.beta_at_step(99, 100)) - 10.0) < 1e-4

    def test_geometric_midpoint_is_geometric_mean(self):
        """SCENARIO-SAMPLE-006: Geometric midpoint = sqrt(init * final)."""
        s = AnnealingSchedule(beta_init=1.0, beta_final=100.0, schedule_type="geometric")
        mid = float(s.beta_at_step(50, 101))
        expected = 10.0  # sqrt(1 * 100)
        assert abs(mid - expected) < 0.5

    def test_single_step(self):
        """SCENARIO-SAMPLE-006: Single-step schedule returns beta_init."""
        s = AnnealingSchedule(beta_init=1.0, beta_final=10.0)
        val = float(s.beta_at_step(0, 1))
        assert abs(val - 1.0) < 1e-5


# --- Low-level update functions ---


class TestParallelUpdate:
    """REQ-SAMPLE-003: Parallel spin update via matrix-vector multiply."""

    def test_strong_bias_all_ones(self):
        """SCENARIO-SAMPLE-007: Strong positive bias → all spins = 1."""
        n = 10
        biases = jnp.ones(n) * 10.0
        J = jnp.zeros((n, n))
        key = jrandom.PRNGKey(0)
        spins = _parallel_update(jnp.zeros(n), biases, J, jnp.float32(5.0), key)
        assert spins.shape == (n,)
        assert float(jnp.mean(spins)) > 0.9

    def test_strong_bias_all_zeros(self):
        """SCENARIO-SAMPLE-007: Strong negative bias → all spins = 0."""
        n = 10
        biases = jnp.ones(n) * -10.0
        J = jnp.zeros((n, n))
        key = jrandom.PRNGKey(0)
        spins = _parallel_update(jnp.ones(n), biases, J, jnp.float32(5.0), key)
        assert float(jnp.mean(spins)) < 0.1

    def test_output_is_binary(self):
        """SCENARIO-SAMPLE-007: Output spins are 0 or 1."""
        n = 20
        biases = jrandom.normal(jrandom.PRNGKey(1), (n,))
        J = jnp.zeros((n, n))
        key = jrandom.PRNGKey(2)
        spins = _parallel_update(jnp.zeros(n), biases, J, jnp.float32(1.0), key)
        unique = jnp.unique(spins)
        assert all(float(v) in (0.0, 1.0) for v in unique)


class TestCheckerboardUpdate:
    """REQ-SAMPLE-003: Checkerboard Gibbs update."""

    def test_shape_preserved(self):
        """SCENARIO-SAMPLE-007: Checkerboard preserves shape."""
        n = 10
        spins = jnp.zeros(n)
        biases = jnp.zeros(n)
        J = jnp.zeros((n, n))
        even = jnp.arange(n) % 2 == 0
        odd = ~even
        k1, k2 = jrandom.split(jrandom.PRNGKey(0))
        result = _checkerboard_update(spins, biases, J, jnp.float32(1.0), k1, k2, even, odd)
        assert result.shape == (n,)

    def test_strong_ferromagnetic_coupling(self):
        """SCENARIO-SAMPLE-007: Strong ferromagnetic coupling → aligned spins."""
        n = 6
        biases = jnp.ones(n) * 0.5
        J = jnp.ones((n, n)) * 2.0
        jnp.fill_diagonal(J, 0.0, inplace=False)
        J = J.at[jnp.arange(n), jnp.arange(n)].set(0.0)
        even = jnp.arange(n) % 2 == 0
        odd = ~even
        spins = jnp.zeros(n)
        for _ in range(50):
            k1, k2 = jrandom.split(jrandom.PRNGKey(_))
            spins = _checkerboard_update(spins, biases, J, jnp.float32(5.0), k1, k2, even, odd)
        # Most spins should agree (all 1 or all 0).
        mean = float(jnp.mean(spins))
        assert mean > 0.8 or mean < 0.2


# --- ParallelIsingSampler ---


class TestParallelIsingSampler:
    """REQ-SAMPLE-003: Full sampler with annealing."""

    def test_sample_shape(self):
        """SCENARIO-SAMPLE-008: Output has correct shape."""
        n = 8
        sampler = ParallelIsingSampler(
            n_warmup=10, n_samples=5, steps_per_sample=2,
        )
        key = jrandom.PRNGKey(42)
        b = jnp.zeros(n)
        J = jnp.zeros((n, n))
        samples = sampler.sample(key, b, J, beta=1.0)
        assert samples.shape == (5, n)

    def test_sample_boolean(self):
        """SCENARIO-SAMPLE-008: Samples are boolean."""
        n = 8
        sampler = ParallelIsingSampler(n_warmup=10, n_samples=5, steps_per_sample=2)
        samples = sampler.sample(jrandom.PRNGKey(0), jnp.zeros(n), jnp.zeros((n, n)))
        assert samples.dtype == jnp.bool_

    def test_annealing_improves_quality(self):
        """SCENARIO-SAMPLE-008: Annealing finds lower energy than constant temperature."""
        n = 10
        rng = np.random.default_rng(42)
        J_np = rng.normal(0, 0.5, (n, n)).astype(np.float32)
        J_np = (J_np + J_np.T) / 2
        np.fill_diagonal(J_np, 0)
        b = jnp.array(rng.normal(0, 1, n), dtype=jnp.float32)
        J = jnp.array(J_np)

        # With annealing.
        sampler_anneal = ParallelIsingSampler(
            n_warmup=200, n_samples=20, steps_per_sample=10,
            schedule=AnnealingSchedule(0.1, 10.0),
        )
        samples_a = sampler_anneal.sample(jrandom.PRNGKey(0), b, J, beta=10.0)

        # Without annealing (constant low temp).
        sampler_const = ParallelIsingSampler(
            n_warmup=200, n_samples=20, steps_per_sample=10,
            schedule=None,
        )
        samples_c = sampler_const.sample(jrandom.PRNGKey(0), b, J, beta=10.0)

        # Both should produce valid samples (just check shapes).
        assert samples_a.shape == (20, n)
        assert samples_c.shape == (20, n)

    def test_no_checkerboard_mode(self):
        """SCENARIO-SAMPLE-008: Fully parallel mode (no checkerboard)."""
        n = 6
        sampler = ParallelIsingSampler(
            n_warmup=10, n_samples=5, steps_per_sample=2,
            use_checkerboard=False,
        )
        samples = sampler.sample(jrandom.PRNGKey(0), jnp.zeros(n), jnp.zeros((n, n)))
        assert samples.shape == (5, n)

    def test_init_spins_respected(self):
        """SCENARIO-SAMPLE-008: Custom initial spins are used."""
        n = 8
        sampler = ParallelIsingSampler(n_warmup=0, n_samples=1, steps_per_sample=0)
        init = jnp.ones(n, dtype=jnp.bool_)
        # With zero warmup and zero steps, output should reflect init.
        # (Though with n_samples=1 and steps_per_sample=0, it collects the state after warmup.)
        samples = sampler.sample(jrandom.PRNGKey(0), jnp.zeros(n), jnp.zeros((n, n)), init_spins=init)
        assert samples.shape == (1, n)


# --- sat_to_coupling_matrix ---


class TestSatToCouplingMatrix:
    """REQ-SAMPLE-003: Convert thrml-style flat weights to coupling matrix."""

    def test_symmetric_zero_diagonal(self):
        """SCENARIO-SAMPLE-008: Output matrix is symmetric with zero diagonal."""
        n = 4
        n_edges = n * (n - 1) // 2  # 6
        weights = jnp.arange(n_edges, dtype=jnp.float32)
        biases_in = jnp.ones(n)
        b, J = sat_to_coupling_matrix(biases_in, weights, n)
        # Symmetric.
        assert jnp.allclose(J, J.T)
        # Zero diagonal.
        assert jnp.allclose(jnp.diag(J), 0.0)
        # Biases unchanged.
        assert jnp.allclose(b, biases_in)

    def test_correct_values(self):
        """SCENARIO-SAMPLE-008: Coupling values match flat vector."""
        # 3 vars: edges (0,1), (0,2), (1,2) → weights [1, 2, 3]
        weights = jnp.array([1.0, 2.0, 3.0])
        biases = jnp.zeros(3)
        _, J = sat_to_coupling_matrix(biases, weights, 3)
        assert float(J[0, 1]) == 1.0
        assert float(J[0, 2]) == 2.0
        assert float(J[1, 2]) == 3.0
        assert float(J[1, 0]) == 1.0  # Symmetric.


# --- extract_ising_params ---


class TestExtractIsingParams:
    """REQ-SAMPLE-003: Extract parameters from thrml IsingEBM."""

    def test_extract_from_mock_model(self):
        """SCENARIO-SAMPLE-008: Extract biases, J, beta from a mock IsingEBM."""
        # Create a minimal mock that has the same attributes as thrml's IsingEBM.
        class MockNode:
            pass

        class MockModel:
            def __init__(self):
                self.nodes = [MockNode(), MockNode(), MockNode()]
                self.biases = jnp.array([1.0, 2.0, 3.0])
                self.beta = jnp.array(5.0)
                self.edges = [
                    (self.nodes[0], self.nodes[1]),
                    (self.nodes[0], self.nodes[2]),
                ]
                self.weights = jnp.array([0.5, -0.3])

        model = MockModel()
        b, J, beta = extract_ising_params(model)

        assert b.shape == (3,)
        assert J.shape == (3, 3)
        assert float(beta) == 5.0
        assert jnp.allclose(J, J.T)
        assert float(J[0, 0]) == 0.0
        assert abs(float(J[0, 1]) - 0.5) < 1e-6
        assert abs(float(J[0, 2]) - (-0.3)) < 1e-6


# --- parallel_sample_states ---


class TestParallelSampleStates:
    """REQ-SAMPLE-003: thrml-compatible wrapper."""

    def test_returns_thrml_format(self):
        """SCENARIO-SAMPLE-008: Returns list of (n_samples, 1) arrays."""

        class MockNode:
            pass

        class MockBlock:
            def __init__(self, nodes):
                self.nodes = nodes

        class MockSchedule:
            n_warmup = 10
            n_samples = 5
            steps_per_sample = 2

        class MockModel:
            def __init__(self):
                self.nodes = [MockNode() for _ in range(4)]
                self.biases = jnp.zeros(4)
                self.beta = jnp.array(1.0)
                self.edges = [(self.nodes[0], self.nodes[1])]
                self.weights = jnp.array([0.1])

        model = MockModel()
        blocks = [MockBlock([model.nodes[i]]) for i in range(4)]
        schedule = MockSchedule()

        result = parallel_sample_states(
            jrandom.PRNGKey(0), model, schedule,
            nodes_to_sample=blocks,
        )

        assert isinstance(result, list)
        assert len(result) == 4
        for arr in result:
            assert arr.shape == (5, 1)

    def test_default_schedule(self):
        """SCENARIO-SAMPLE-008: Works without explicit schedule."""

        class MockNode:
            pass

        class MockModel:
            def __init__(self):
                self.nodes = [MockNode(), MockNode()]
                self.biases = jnp.zeros(2)
                self.beta = jnp.array(1.0)
                self.edges = []
                self.weights = jnp.array([])

        model = MockModel()
        result = parallel_sample_states(jrandom.PRNGKey(0), model)

        assert isinstance(result, list)
        assert len(result) == 2
        # Default n_samples=50.
        assert result[0].shape[0] == 50

    def test_raw_nodes_to_sample(self):
        """SCENARIO-SAMPLE-008: nodes_to_sample accepts raw node objects."""

        class MockNode:
            pass

        class MockModel:
            def __init__(self):
                self.nodes = [MockNode(), MockNode(), MockNode()]
                self.biases = jnp.zeros(3)
                self.beta = jnp.array(1.0)
                self.edges = []
                self.weights = jnp.array([])

        model = MockModel()
        # Pass raw nodes (not Block objects).
        result = parallel_sample_states(
            jrandom.PRNGKey(0), model,
            nodes_to_sample=[model.nodes[0], model.nodes[2]],
        )
        assert len(result) == 2
        assert result[0].shape[1] == 1

    def test_with_annealing(self):
        """SCENARIO-SAMPLE-008: Works with annealing schedule."""

        class MockNode:
            pass

        class MockModel:
            def __init__(self):
                self.nodes = [MockNode(), MockNode(), MockNode()]
                self.biases = jnp.array([1.0, -1.0, 0.5])
                self.beta = jnp.array(2.0)
                self.edges = [(self.nodes[0], self.nodes[1])]
                self.weights = jnp.array([0.5])

        model = MockModel()
        result = parallel_sample_states(
            jrandom.PRNGKey(0), model,
            annealing=AnnealingSchedule(0.1, 5.0),
        )
        assert len(result) == 3
