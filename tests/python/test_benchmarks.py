"""Tests for analytical benchmark energy functions — JAX implementation.

Verifies that each benchmark:
1. Produces known energy at the global minimum
2. Has zero gradient at the global minimum
3. Has higher energy away from the minimum
4. Conforms to the EnergyFunction protocol
5. Matches expected symmetry properties

Spec coverage: REQ-AUTO-001
"""

import jax.numpy as jnp
import pytest
from carnot.benchmarks import (
    Ackley,
    DoubleWell,
    GaussianMixture,
    Rastrigin,
    Rosenbrock,
)
from carnot.benchmarks.functions import get_standard_benchmarks
from carnot.core.energy import EnergyFunction


class TestDoubleWell:
    """Tests for DoubleWell benchmark — REQ-AUTO-001."""

    def test_energy_at_minimum(self) -> None:
        """REQ-AUTO-001: energy = 0 at minimum [1, 0, ...]."""
        dw = DoubleWell(dim=3)
        info = dw.info()
        e = dw.energy(info.global_min_location)
        assert jnp.abs(e) < 1e-6

    def test_symmetry(self) -> None:
        """REQ-AUTO-001: two symmetric minima at x[0] = +/-1."""
        dw = DoubleWell(dim=2)
        x_pos = jnp.array([1.0, 0.0])
        x_neg = jnp.array([-1.0, 0.0])
        assert jnp.abs(dw.energy(x_pos) - dw.energy(x_neg)) < 1e-6
        assert dw.energy(x_pos) < 1e-6

    def test_barrier(self) -> None:
        """REQ-AUTO-001: barrier at x[0]=0 has energy 1.0."""
        dw = DoubleWell(dim=2)
        e = dw.energy(jnp.array([0.0, 0.0]))
        assert jnp.abs(e - 1.0) < 1e-6

    def test_gradient_at_minimum(self) -> None:
        """REQ-AUTO-001: gradient ~0 at minimum."""
        dw = DoubleWell(dim=2)
        grad = dw.grad_energy(jnp.array([1.0, 0.0]))
        assert jnp.all(jnp.abs(grad) < 1e-4)

    def test_interface(self) -> None:
        """REQ-AUTO-001: conforms to EnergyFunction protocol."""
        dw = DoubleWell(dim=2)
        assert isinstance(dw, EnergyFunction)
        assert dw.input_dim == 2

    def test_batch(self) -> None:
        """REQ-AUTO-001: batch energy works."""
        dw = DoubleWell(dim=2)
        xs = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
        energies = dw.energy_batch(xs)
        assert energies.shape == (3,)

    def test_invalid_dim(self) -> None:
        """REQ-AUTO-001: dim < 1 raises error."""
        with pytest.raises(ValueError):
            DoubleWell(dim=0)


class TestRosenbrock:
    """Tests for Rosenbrock benchmark — REQ-AUTO-001."""

    def test_energy_at_minimum(self) -> None:
        """REQ-AUTO-001: energy = 0 at [1, 1, ..., 1]."""
        rb = Rosenbrock(dim=4)
        e = rb.energy(jnp.ones(4))
        assert jnp.abs(e) < 1e-6

    def test_energy_at_origin(self) -> None:
        """REQ-AUTO-001: energy = 1.0 at [0, 0]."""
        rb = Rosenbrock(dim=2)
        e = rb.energy(jnp.zeros(2))
        assert jnp.abs(e - 1.0) < 1e-6

    def test_gradient_at_minimum(self) -> None:
        """REQ-AUTO-001: gradient ~0 at minimum."""
        rb = Rosenbrock(dim=3)
        grad = rb.grad_energy(jnp.ones(3))
        assert jnp.all(jnp.abs(grad) < 1e-3)

    def test_interface(self) -> None:
        """REQ-AUTO-001: conforms to EnergyFunction protocol."""
        rb = Rosenbrock(dim=2)
        assert isinstance(rb, EnergyFunction)
        assert rb.input_dim == 2

    def test_invalid_dim(self) -> None:
        """REQ-AUTO-001: dim < 2 raises error."""
        with pytest.raises(ValueError):
            Rosenbrock(dim=1)


class TestAckley:
    """Tests for Ackley benchmark — REQ-AUTO-001."""

    def test_energy_at_minimum(self) -> None:
        """REQ-AUTO-001: energy ~0 at origin."""
        ack = Ackley(dim=3)
        e = ack.energy(jnp.zeros(3))
        assert jnp.abs(e) < 1e-4

    def test_energy_away_from_min(self) -> None:
        """REQ-AUTO-001: energy > 0 away from origin."""
        ack = Ackley(dim=2)
        e = ack.energy(jnp.ones(2))
        assert e > 0.0

    def test_gradient_at_minimum(self) -> None:
        """REQ-AUTO-001: gradient ~0 at origin."""
        ack = Ackley(dim=2)
        grad = ack.grad_energy(jnp.zeros(2))
        assert jnp.all(jnp.abs(grad) < 1e-3)

    def test_interface(self) -> None:
        """REQ-AUTO-001: conforms to EnergyFunction protocol."""
        ack = Ackley(dim=2)
        assert isinstance(ack, EnergyFunction)
        assert ack.input_dim == 2

    def test_invalid_dim(self) -> None:
        """REQ-AUTO-001: dim < 1 raises error."""
        with pytest.raises(ValueError):
            Ackley(dim=0)


class TestRastrigin:
    """Tests for Rastrigin benchmark — REQ-AUTO-001."""

    def test_energy_at_minimum(self) -> None:
        """REQ-AUTO-001: energy = 0 at origin."""
        ras = Rastrigin(dim=3)
        e = ras.energy(jnp.zeros(3))
        assert jnp.abs(e) < 1e-4

    def test_local_minimum(self) -> None:
        """REQ-AUTO-001: local minimum at x=[1] has energy 1.0."""
        ras = Rastrigin(dim=1)
        e = ras.energy(jnp.array([1.0]))
        assert jnp.abs(e - 1.0) < 1e-4

    def test_gradient_at_minimum(self) -> None:
        """REQ-AUTO-001: gradient ~0 at origin."""
        ras = Rastrigin(dim=2)
        grad = ras.grad_energy(jnp.zeros(2))
        assert jnp.all(jnp.abs(grad) < 1e-3)

    def test_interface(self) -> None:
        """REQ-AUTO-001: conforms to EnergyFunction protocol."""
        ras = Rastrigin(dim=2)
        assert isinstance(ras, EnergyFunction)
        assert ras.input_dim == 2

    def test_invalid_dim(self) -> None:
        """REQ-AUTO-001: dim < 1 raises error."""
        with pytest.raises(ValueError):
            Rastrigin(dim=0)


class TestGaussianMixture:
    """Tests for GaussianMixture benchmark — REQ-AUTO-001."""

    def test_two_modes(self) -> None:
        """REQ-AUTO-001: two_modes creates correct 1D mixture."""
        gmm = GaussianMixture.two_modes(4.0)
        assert gmm.input_dim == 1
        assert len(gmm.means) == 2

    def test_symmetric_modes(self) -> None:
        """REQ-AUTO-001: energy equal at both modes."""
        gmm = GaussianMixture.two_modes(4.0)
        e1 = gmm.energy(jnp.array([-2.0]))
        e2 = gmm.energy(jnp.array([2.0]))
        assert jnp.abs(e1 - e2) < 1e-4

    def test_modes_lower_than_between(self) -> None:
        """REQ-AUTO-001: modes have lower energy than midpoint."""
        gmm = GaussianMixture.two_modes(4.0)
        e_mode = gmm.energy(jnp.array([-2.0]))
        e_mid = gmm.energy(jnp.array([0.0]))
        assert e_mode < e_mid

    def test_multidimensional(self) -> None:
        """REQ-AUTO-001: works in multiple dimensions."""
        gmm = GaussianMixture(
            dim=3,
            means=[jnp.zeros(3), jnp.ones(3)],
            variances=[1.0, 1.0],
            weights=[0.5, 0.5],
        )
        assert gmm.input_dim == 3
        e = gmm.energy(jnp.zeros(3))
        assert jnp.isfinite(e)

    def test_interface(self) -> None:
        """REQ-AUTO-001: conforms to EnergyFunction protocol."""
        gmm = GaussianMixture.two_modes()
        assert isinstance(gmm, EnergyFunction)

    def test_gradient_finite(self) -> None:
        """REQ-AUTO-001: gradient is finite."""
        gmm = GaussianMixture.two_modes(4.0)
        grad = gmm.grad_energy(jnp.array([0.0]))
        assert jnp.all(jnp.isfinite(grad))

    def test_invalid_dim(self) -> None:
        """REQ-AUTO-001: dim < 1 raises error."""
        with pytest.raises(ValueError):
            GaussianMixture(dim=0, means=[], variances=[], weights=[])

    def test_mismatched_lengths(self) -> None:
        """REQ-AUTO-001: mismatched means/variances raises error."""
        with pytest.raises(ValueError):
            GaussianMixture(
                dim=1,
                means=[jnp.array([0.0])],
                variances=[1.0, 2.0],
                weights=[1.0],
            )


class TestGetStandardBenchmarks:
    """Tests for the convenience function — REQ-AUTO-001."""

    def test_returns_all_five(self) -> None:
        """REQ-AUTO-001: get_standard_benchmarks returns all 5 benchmarks."""
        benchmarks = get_standard_benchmarks(dim=2)
        assert len(benchmarks) == 5
        assert set(benchmarks.keys()) == {
            "double_well", "rosenbrock", "ackley", "rastrigin", "gaussian_mixture"
        }

    def test_all_have_info(self) -> None:
        """REQ-AUTO-001: each benchmark has valid info."""
        for name, (fn, info) in get_standard_benchmarks(dim=2).items():
            assert info.name == name
            assert info.input_dim >= 1
            assert jnp.isfinite(fn.energy(info.global_min_location))

    def test_1d_benchmarks(self) -> None:
        """REQ-AUTO-001: works with dim=1 (Rosenbrock auto-bumps to 2)."""
        benchmarks = get_standard_benchmarks(dim=1)
        # Rosenbrock requires dim>=2, so it should auto-bump
        _, info = benchmarks["rosenbrock"]
        assert info.input_dim == 2
