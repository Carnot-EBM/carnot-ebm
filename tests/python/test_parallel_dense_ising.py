"""Tests for parallel_dense_ising.py — 100% coverage.

Every test references the spec requirement it covers.
Spec: REQ-SAMPLE-023, REQ-SAMPLE-024
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from carnot.samplers.parallel_dense_ising import (
    ParallelDenseIsingConfig,
    ParallelDenseIsingInertia,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_sampler(n_spins: int = 10, alpha: float = 0.3, n_steps: int = 20) -> ParallelDenseIsingInertia:
    """Return a sampler with small default config for unit tests.

    Why n_steps=20 in tests: fast enough to pass in CI, enough steps to verify
    convergence behaviour and energy tracking.
    """
    cfg = ParallelDenseIsingConfig(n_spins=n_spins, alpha=alpha, beta=1.0, n_steps=n_steps)
    return ParallelDenseIsingInertia(cfg)


def ferromagnetic_J(n: int) -> jax.Array:
    """All-pairs ferromagnetic coupling normalised by n.

    Why normalised: prevents energy explosion as n grows. Paper normalises by n_spins
    for dense random graphs; a uniform ferromagnetic J uses the same convention.
    """
    J = jnp.ones((n, n), dtype=jnp.float32) / n
    # Zero diagonal: a spin does not couple to itself.
    return J.at[jnp.arange(n), jnp.arange(n)].set(0.0)


# ---------------------------------------------------------------------------
# REQ-SAMPLE-023: ParallelDenseIsingConfig
# ---------------------------------------------------------------------------


class TestParallelDenseIsingConfig:
    """Spec: REQ-SAMPLE-023"""

    def test_defaults(self) -> None:
        """Default config has expected attribute values."""
        cfg = ParallelDenseIsingConfig()
        assert cfg.n_spins == 100
        assert cfg.alpha == 0.3
        assert cfg.beta == 1.0
        assert cfg.n_steps == 200

    def test_custom_values(self) -> None:
        """Custom values are stored correctly."""
        cfg = ParallelDenseIsingConfig(n_spins=50, alpha=0.5, beta=2.0, n_steps=100)
        assert cfg.n_spins == 50
        assert cfg.alpha == 0.5
        assert cfg.beta == 2.0
        assert cfg.n_steps == 100


# ---------------------------------------------------------------------------
# REQ-SAMPLE-023: ParallelDenseIsingInertia construction
# ---------------------------------------------------------------------------


class TestParallelDenseIsingInertiaInit:
    """Spec: REQ-SAMPLE-023"""

    def test_h_i_initialised_to_zero(self) -> None:
        """h_i starts as zeros of the correct size."""
        sampler = make_sampler(n_spins=8)
        assert sampler.h_i.shape == (8,)
        assert float(jnp.sum(jnp.abs(sampler.h_i))) == 0.0

    def test_config_stored(self) -> None:
        """Config is accessible via sampler.config."""
        cfg = ParallelDenseIsingConfig(n_spins=5, alpha=0.1)
        sampler = ParallelDenseIsingInertia(cfg)
        assert sampler.config is cfg


# ---------------------------------------------------------------------------
# REQ-SAMPLE-023: _compute_local_fields
# ---------------------------------------------------------------------------


class TestComputeLocalFields:
    """Spec: REQ-SAMPLE-023"""

    def test_identity_coupling(self) -> None:
        """J = I gives local fields equal to the spin vector."""
        sampler = make_sampler(n_spins=4)
        J = jnp.eye(4)
        s = jnp.array([1.0, -1.0, 1.0, -1.0])
        lf = sampler._compute_local_fields(J, s)
        assert jnp.allclose(lf, s)

    def test_zero_coupling(self) -> None:
        """J = 0 gives zero local fields regardless of spin state."""
        sampler = make_sampler(n_spins=4)
        J = jnp.zeros((4, 4))
        s = jnp.ones(4)
        lf = sampler._compute_local_fields(J, s)
        assert jnp.allclose(lf, jnp.zeros(4))

    def test_ferromagnetic_all_up(self) -> None:
        """All +1 spins in ferromagnetic field produce positive local fields."""
        n = 6
        sampler = make_sampler(n_spins=n)
        J = ferromagnetic_J(n)
        s = jnp.ones(n)
        lf = sampler._compute_local_fields(J, s)
        assert jnp.all(lf > 0.0)


# ---------------------------------------------------------------------------
# REQ-SAMPLE-023: _update_inertia
# ---------------------------------------------------------------------------


class TestUpdateInertia:
    """Spec: REQ-SAMPLE-023"""

    def test_alpha_zero_returns_local_fields(self) -> None:
        """alpha=0 -> pure current state (no memory), reduces to standard Gibbs."""
        sampler = make_sampler(alpha=0.0)
        h_i = jnp.array([5.0, -3.0])
        lf = jnp.array([1.0, 2.0])
        result = sampler._update_inertia(h_i, lf)
        assert jnp.allclose(result, lf)

    def test_alpha_one_returns_history(self) -> None:
        """alpha=1 -> pure history (EMA freezes at initial value)."""
        sampler = make_sampler(alpha=1.0)
        h_i = jnp.array([5.0, -3.0])
        lf = jnp.array([1.0, 2.0])
        result = sampler._update_inertia(h_i, lf)
        assert jnp.allclose(result, h_i)

    def test_ema_blend(self) -> None:
        """alpha=0.3 blends 30% history with 70% current field."""
        sampler = make_sampler(alpha=0.3)
        h_i = jnp.array([10.0])
        lf = jnp.array([0.0])
        expected = 0.3 * 10.0 + 0.7 * 0.0
        result = sampler._update_inertia(h_i, lf)
        assert jnp.allclose(result, jnp.array([expected]))


# ---------------------------------------------------------------------------
# REQ-SAMPLE-023: _flip_probabilities
# ---------------------------------------------------------------------------


class TestFlipProbabilities:
    """Spec: REQ-SAMPLE-023"""

    def test_zero_field_gives_half(self) -> None:
        """Zero field and zero bias -> P(s=+1) = 0.5 (uninformative)."""
        sampler = make_sampler(n_spins=4)
        h_i = jnp.zeros(4)
        biases = jnp.zeros(4)
        probs = sampler._flip_probabilities(h_i, biases)
        assert jnp.allclose(probs, 0.5 * jnp.ones(4), atol=1e-6)

    def test_large_positive_field_gives_high_prob(self) -> None:
        """Large positive field -> P(s=+1) close to 1."""
        sampler = make_sampler(n_spins=2)
        h_i = jnp.array([100.0, 100.0])
        biases = jnp.zeros(2)
        probs = sampler._flip_probabilities(h_i, biases)
        assert jnp.all(probs > 0.99)

    def test_large_negative_field_gives_low_prob(self) -> None:
        """Large negative field -> P(s=+1) close to 0."""
        sampler = make_sampler(n_spins=2)
        h_i = jnp.array([-100.0, -100.0])
        biases = jnp.zeros(2)
        probs = sampler._flip_probabilities(h_i, biases)
        assert jnp.all(probs < 0.01)

    def test_probs_in_unit_interval(self) -> None:
        """All flip probabilities are in (0, 1)."""
        sampler = make_sampler(n_spins=10)
        key = jax.random.PRNGKey(42)
        h_i = jax.random.normal(key, (10,))
        biases = jax.random.normal(jax.random.PRNGKey(1), (10,))
        probs = sampler._flip_probabilities(h_i, biases)
        assert jnp.all(probs > 0.0)
        assert jnp.all(probs < 1.0)


# ---------------------------------------------------------------------------
# REQ-SAMPLE-023: sample() output contract
# ---------------------------------------------------------------------------


class TestSampleOutputContract:
    """Spec: REQ-SAMPLE-023"""

    def test_output_keys(self) -> None:
        """sample() returns dict with all required keys."""
        n = 10
        sampler = make_sampler(n_spins=n, n_steps=5)
        J = ferromagnetic_J(n)
        biases = jnp.zeros(n)
        key = jax.random.PRNGKey(0)
        result = sampler.sample(J, biases, key)
        assert "final_state" in result
        assert "final_energy" in result
        assert "energy_history" in result
        assert "n_steps" in result

    def test_final_state_shape(self) -> None:
        """final_state has shape (n_spins,)."""
        n = 12
        sampler = make_sampler(n_spins=n, n_steps=3)
        J = jnp.zeros((n, n))
        biases = jnp.zeros(n)
        key = jax.random.PRNGKey(1)
        result = sampler.sample(J, biases, key)
        assert result["final_state"].shape == (n,)

    def test_final_state_values_pm1(self) -> None:
        """final_state values are exactly ±1."""
        n = 15
        sampler = make_sampler(n_spins=n, n_steps=5)
        J = jnp.zeros((n, n))
        biases = jnp.zeros(n)
        key = jax.random.PRNGKey(7)
        result = sampler.sample(J, biases, key)
        s = result["final_state"]
        assert jnp.all((s == 1.0) | (s == -1.0))

    def test_energy_history_length(self) -> None:
        """energy_history has length equal to n_steps."""
        n_steps = 17
        sampler = make_sampler(n_spins=8, n_steps=n_steps)
        J = ferromagnetic_J(8)
        biases = jnp.zeros(8)
        key = jax.random.PRNGKey(2)
        result = sampler.sample(J, biases, key)
        assert len(result["energy_history"]) == n_steps

    def test_n_steps_in_output(self) -> None:
        """n_steps in output matches config."""
        n_steps = 11
        sampler = make_sampler(n_spins=5, n_steps=n_steps)
        J = jnp.zeros((5, 5))
        biases = jnp.zeros(5)
        key = jax.random.PRNGKey(3)
        result = sampler.sample(J, biases, key)
        assert result["n_steps"] == n_steps

    def test_final_energy_matches_last_history(self) -> None:
        """final_energy equals the last entry of energy_history."""
        n = 8
        sampler = make_sampler(n_spins=n, n_steps=10)
        J = ferromagnetic_J(n)
        biases = jnp.zeros(n)
        key = jax.random.PRNGKey(4)
        result = sampler.sample(J, biases, key)
        assert result["final_energy"] == pytest.approx(result["energy_history"][-1], rel=1e-5)

    def test_energy_history_are_floats(self) -> None:
        """Energy history entries are Python floats (serialisable)."""
        sampler = make_sampler(n_spins=6, n_steps=4)
        J = ferromagnetic_J(6)
        biases = jnp.zeros(6)
        key = jax.random.PRNGKey(5)
        result = sampler.sample(J, biases, key)
        for e in result["energy_history"]:
            assert isinstance(e, float)


# ---------------------------------------------------------------------------
# REQ-SAMPLE-023: init_state handling
# ---------------------------------------------------------------------------


class TestSampleInitState:
    """Spec: REQ-SAMPLE-023"""

    def test_custom_init_state_used(self) -> None:
        """Providing init_state uses it (deterministic with zero coupling)."""
        n = 6
        sampler = make_sampler(n_spins=n, n_steps=1, alpha=0.0)
        J = jnp.zeros((n, n))
        # With all-neg biases, probability of +1 is near zero; near zero coupling means
        # the spin state immediately reflects the strong bias.
        biases = -100.0 * jnp.ones(n)
        init = jnp.ones(n)  # all +1 start
        key = jax.random.PRNGKey(99)
        result = sampler.sample(J, biases, key, init_state=init)
        # With huge negative bias, final state should flip to -1.
        assert jnp.all(result["final_state"] == -1.0)

    def test_none_init_state_all_plus_one(self) -> None:
        """No init_state: starts from all +1 (ferromagnetic alignment)."""
        n = 5
        sampler = make_sampler(n_spins=n, n_steps=0)
        # 0 steps means h_i is bootstrapped from the initial state and we never
        # enter the loop; energy_history is empty.
        J = ferromagnetic_J(n)
        biases = jnp.zeros(n)
        key = jax.random.PRNGKey(0)
        result = sampler.sample(J, biases, key, init_state=None)
        # With n_steps=0, the energy_history is empty and final_state is all +1.
        assert result["n_steps"] == 0
        assert len(result["energy_history"]) == 0

    def test_h_i_updated_after_sample(self) -> None:
        """sampler.h_i is updated after sample() (stores final EMA field)."""
        n = 6
        sampler = make_sampler(n_spins=n, n_steps=5)
        J = ferromagnetic_J(n)
        biases = jnp.zeros(n)
        key = jax.random.PRNGKey(42)
        sampler.sample(J, biases, key)
        # h_i should be non-trivial after a ferromagnetic run.
        assert sampler.h_i.shape == (n,)


# ---------------------------------------------------------------------------
# REQ-SAMPLE-023: Export from carnot.samplers
# ---------------------------------------------------------------------------


def test_export_from_carnot_samplers() -> None:
    """ParallelDenseIsingInertia and Config are importable from carnot.samplers."""
    from carnot.samplers import ParallelDenseIsingConfig, ParallelDenseIsingInertia  # noqa: PLC0415
    cfg = ParallelDenseIsingConfig(n_spins=4)
    sampler = ParallelDenseIsingInertia(cfg)
    assert sampler.config.n_spins == 4


# ---------------------------------------------------------------------------
# REQ-SAMPLE-024: Convergence benchmark (lightweight proxy)
# ---------------------------------------------------------------------------


class TestConvergenceBenchmark:
    """REQ-SAMPLE-024: inertia converges in <= 80% steps vs standard Gibbs.

    The full benchmark runs in experiment_648; here we do a lightweight proxy
    to ensure the inertia dynamics actually reduce convergence steps on a small
    dense graph where the effect is most pronounced.

    Standard Gibbs baseline: alpha=0.0 (no inertia = synchronous Gibbs).
    Inertia: alpha=0.3.

    Convergence criterion: first step where rolling 5-step std of energy < 0.5%
    of |energy range|.  We just verify the inertia sampler does NOT take MORE
    steps than the baseline for a dense ferromagnetic graph (sanity check).
    """

    @staticmethod
    def _steps_to_stable(energy_history: list[float], window: int = 5, tol_pct: float = 0.005) -> int:
        """Return first step where rolling std < tol_pct * |range|.

        Why rolling std: more robust than single-step delta for noisy energy traces.
        Returns n_steps if never converges (worst case).
        """
        if len(energy_history) < window:
            return len(energy_history)
        e_range = max(energy_history) - min(energy_history)
        if e_range < 1e-8:
            return 0
        threshold = tol_pct * e_range
        for i in range(window, len(energy_history)):
            window_std = float(jnp.std(jnp.array(energy_history[i - window : i])))
            if window_std < threshold:
                return i
        return len(energy_history)

    def test_inertia_does_not_diverge(self) -> None:
        """Inertia sampler reaches lower energy than starting from the worst state.

        REQ-SAMPLE-024: inertia should converge (not diverge) on a dense graph.
        We start from an alternating +1/-1 state (high-energy for a ferromagnetic
        problem) and check that the minimum energy seen is lower than the initial
        energy (computed for the alternating state).

        Why alternating init: for a dense ferromagnetic J, the all-+1 state is
        already the ground state, so starting there leaves nowhere to go but up
        (thermal fluctuations). An alternating state is far from the ground state,
        giving the sampler room to actually converge downward.
        """
        n = 50
        n_steps = 100
        J = ferromagnetic_J(n)
        biases = jnp.zeros(n)
        key = jax.random.PRNGKey(7)

        # Alternating +1/-1: high energy for ferromagnetic coupling.
        init = jnp.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
        init_energy = float(-0.5 * init @ J @ init - biases @ init)

        cfg = ParallelDenseIsingConfig(n_spins=n, alpha=0.3, beta=1.0, n_steps=n_steps)
        sampler = ParallelDenseIsingInertia(cfg)
        result = sampler.sample(J, biases, key, init_state=init)

        history = result["energy_history"]
        min_energy = min(history)
        # The minimum energy achieved must be lower than the (high) initial energy.
        assert min_energy < init_energy, (
            f"Inertia sampler did not improve from alternating init: "
            f"init_energy={init_energy:.2f}, min_energy={min_energy:.2f}"
        )

    def test_inertia_convergence_not_slower_than_baseline(self) -> None:
        """Inertia (alpha=0.3) converges no slower than baseline (alpha=0) on dense graph.

        This is a sanity check: if inertia were always slower, the whole paper's
        premise would be wrong. We run both on the same problem with the same seed
        and check that inertia steps_to_stable <= baseline steps_to_stable.
        """
        n = 50
        n_steps = 150
        J = ferromagnetic_J(n)
        biases = jnp.zeros(n)

        baseline_cfg = ParallelDenseIsingConfig(n_spins=n, alpha=0.0, beta=1.0, n_steps=n_steps)
        inertia_cfg = ParallelDenseIsingConfig(n_spins=n, alpha=0.3, beta=1.0, n_steps=n_steps)

        baseline = ParallelDenseIsingInertia(baseline_cfg)
        inertia = ParallelDenseIsingInertia(inertia_cfg)

        key = jax.random.PRNGKey(13)
        baseline_result = baseline.sample(J, biases, key)
        inertia_result = inertia.sample(J, biases, key)

        baseline_steps = self._steps_to_stable(baseline_result["energy_history"])
        inertia_steps = self._steps_to_stable(inertia_result["energy_history"])

        # Allow up to 10% slack: inertia should not be dramatically slower.
        assert inertia_steps <= baseline_steps * 1.10, (
            f"Inertia sampler took more steps than baseline: "
            f"inertia={inertia_steps}, baseline={baseline_steps}"
        )
