"""Tests for repair convergence guarantees (absorbing invariant sets).

Spec coverage: REQ-VERIFY-008
"""

from __future__ import annotations

import jax.numpy as jnp

from carnot.verify.convergence import (
    ConvergenceCertificate,
    certify_repair_convergence,
    compute_absorbing_radius,
    estimate_jacobian_bound,
)


def _quadratic_energy(x: jnp.ndarray) -> jnp.ndarray:
    """E(x) = 0.5 * ||x||^2. Hessian = I, spectral norm = 1."""
    return 0.5 * jnp.sum(x**2)


class _QuadModel:
    """Wrapper with .energy() method."""

    def energy(self, x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.sum(x**2)


class TestEstimateJacobianBound:
    """Tests for Jacobian spectral norm estimation."""

    def test_quadratic_has_unit_hessian(self) -> None:
        """REQ-VERIFY-008: quadratic E=0.5||x||^2 has Hessian=I, norm=1."""
        samples = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        bound = estimate_jacobian_bound(_QuadModel(), samples)
        assert abs(bound - 1.0) < 0.1  # Should be ~1.0

    def test_returns_positive(self) -> None:
        """REQ-VERIFY-008: bound is positive."""
        samples = jnp.ones((3, 2))
        bound = estimate_jacobian_bound(_quadratic_energy, samples)
        assert bound > 0

    def test_works_with_callable(self) -> None:
        """REQ-VERIFY-008: accepts raw callable."""
        samples = jnp.ones((2, 3))
        bound = estimate_jacobian_bound(_quadratic_energy, samples)
        assert bound > 0

    def test_constant_energy_zero_bound(self) -> None:
        """REQ-VERIFY-008: constant energy has zero Jacobian norm."""

        def constant_energy(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.float32(0.0)

        samples = jnp.ones((2, 2))
        bound = estimate_jacobian_bound(constant_energy, samples)
        assert bound == 0.0


class TestComputeAbsorbingRadius:
    """Tests for absorbing radius computation."""

    def test_positive_radius(self) -> None:
        """REQ-VERIFY-008: radius is positive."""
        samples = jnp.ones((2, 2))
        radius = compute_absorbing_radius(_QuadModel(), samples, step_size=0.1)
        assert radius > 0

    def test_larger_step_larger_radius(self) -> None:
        """REQ-VERIFY-008: larger step size gives larger radius."""
        samples = jnp.ones((2, 2))
        r_small = compute_absorbing_radius(_QuadModel(), samples, step_size=0.01)
        r_large = compute_absorbing_radius(_QuadModel(), samples, step_size=0.1)
        assert r_large > r_small

    def test_constant_energy_infinite_radius(self) -> None:
        """REQ-VERIFY-008: constant energy → infinite absorbing radius."""

        def constant_energy(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.float32(0.0)

        samples = jnp.ones((2, 2))
        radius = compute_absorbing_radius(constant_energy, samples, step_size=0.1)
        assert radius == float("inf")

    def test_quadratic_known_radius(self) -> None:
        """REQ-VERIFY-008: for E=0.5||x||^2, L=1, r=step/(2*1)."""
        samples = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        radius = compute_absorbing_radius(_QuadModel(), samples, step_size=0.1)
        # Expected: 0.1 / (2 * 1.0) = 0.05
        assert abs(radius - 0.05) < 0.01


class TestCertifyRepairConvergence:
    """Tests for convergence certification."""

    def test_inside_radius_converges(self) -> None:
        """REQ-VERIFY-008: point inside radius gets converges=True."""
        cert = certify_repair_convergence(
            _QuadModel(),
            x_init=jnp.array([0.01, 0.01]),
            x_min=jnp.zeros(2),
            absorbing_radius=0.1,
        )
        assert cert.converges
        assert cert.margin > 0

    def test_outside_radius_no_guarantee(self) -> None:
        """REQ-VERIFY-008: point outside radius gets converges=False."""
        cert = certify_repair_convergence(
            _QuadModel(),
            x_init=jnp.array([5.0, 5.0]),
            x_min=jnp.zeros(2),
            absorbing_radius=0.1,
        )
        assert not cert.converges
        assert cert.margin < 0

    def test_certificate_fields(self) -> None:
        """REQ-VERIFY-008: certificate has correct fields."""
        cert = ConvergenceCertificate()
        assert cert.converges is False
        assert cert.distance_to_minimum == 0.0
        assert cert.absorbing_radius == 0.0
        assert cert.margin == 0.0
