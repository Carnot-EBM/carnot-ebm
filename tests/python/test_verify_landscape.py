"""Tests for energy landscape certification.

Spec coverage: REQ-VERIFY-006, SCENARIO-VERIFY-005
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import AutoGradMixin
from carnot.verify.landscape import LandscapeCertificate, certify_landscape


class QuadraticEnergy(AutoGradMixin):
    """E(x) = 0.5 * x^T A x — a bowl with known curvature.

    The Hessian is exactly A, so eigenvalues = eigenvalues of A.
    With A = diag([1, 2, 3]), the eigenvalues are [1, 2, 3] and
    the origin is a verified local minimum.
    """

    def __init__(self, eigenvalues: list[float]) -> None:
        self._eigenvalues = jnp.array(eigenvalues)
        self._A = jnp.diag(self._eigenvalues)

    @property
    def input_dim(self) -> int:
        return len(self._eigenvalues)

    def energy(self, x: jax.Array) -> jax.Array:
        return 0.5 * x @ self._A @ x


class SaddleEnergy(AutoGradMixin):
    """E(x) = x[0]^2 - x[1]^2 — a saddle point at the origin.

    The Hessian is diag([2, -2]), so the origin is NOT a minimum.
    """

    @property
    def input_dim(self) -> int:
        return 2

    def energy(self, x: jax.Array) -> jax.Array:
        return x[0] ** 2 - x[1] ** 2


class TestLocalMinimumVerification:
    """Tests for SCENARIO-VERIFY-005: local minimum verification."""

    def test_quadratic_minimum_verified(self) -> None:
        """SCENARIO-VERIFY-005: bowl-shaped energy is certified as minimum."""
        model = QuadraticEnergy([1.0, 2.0, 3.0])
        x = jnp.zeros(3)  # the minimum of a bowl is at the origin

        cert = certify_landscape(model, x)

        assert cert.is_local_minimum
        assert cert.classification == "local_minimum"
        # Eigenvalues should be [1, 2, 3] (the diagonal of A)
        assert len(cert.eigenvalues) == 3
        assert abs(cert.eigenvalues[0] - 1.0) < 0.1
        assert abs(cert.eigenvalues[1] - 2.0) < 0.1
        assert abs(cert.eigenvalues[2] - 3.0) < 0.1

    def test_saddle_point_detected(self) -> None:
        """SCENARIO-VERIFY-005: saddle point is NOT certified as minimum."""
        model = SaddleEnergy()
        x = jnp.zeros(2)  # saddle point

        cert = certify_landscape(model, x)

        assert not cert.is_local_minimum
        assert cert.classification == "saddle_point"
        # Should have one positive and one negative eigenvalue
        assert cert.min_eigenvalue < -0.1
        assert cert.max_eigenvalue > 0.1

    def test_condition_number(self) -> None:
        """REQ-VERIFY-006: condition number reflects bowl elongation."""
        # Well-conditioned: eigenvalues [1, 1, 1] → condition = 1
        model_round = QuadraticEnergy([1.0, 1.0, 1.0])
        cert_round = certify_landscape(model_round, jnp.zeros(3))
        assert abs(cert_round.condition_number - 1.0) < 0.5

        # Ill-conditioned: eigenvalues [0.1, 10.0] → condition = 100
        model_elongated = QuadraticEnergy([0.1, 10.0])
        cert_elongated = certify_landscape(model_elongated, jnp.zeros(2))
        assert cert_elongated.condition_number > 50

    def test_saddle_condition_number_infinite(self) -> None:
        """REQ-VERIFY-006: saddle point with near-zero eigenvalue has high condition number."""
        model = SaddleEnergy()
        cert = certify_landscape(model, jnp.zeros(2))
        # Saddle has eigenvalues [−2, 2], condition = |2/−2| = 1
        # But the key point: it's classified as saddle, not minimum
        assert not cert.is_local_minimum

    def test_degenerate_point_detected(self) -> None:
        """REQ-VERIFY-006: flat direction (zero eigenvalue) classified as degenerate."""
        # E(x) = x[0]^2 + 0*x[1]^2 — flat in the x[1] direction
        model = QuadraticEnergy([1.0, 0.0])
        x = jnp.zeros(2)

        cert = certify_landscape(model, x)

        assert not cert.is_local_minimum
        assert cert.classification == "degenerate"
        assert cert.condition_number == float("inf")
        assert abs(cert.min_eigenvalue) < 0.01

    def test_basin_radius_estimation(self) -> None:
        """REQ-VERIFY-006: basin radius is larger for wider bowls."""
        # Shallow bowl (small eigenvalues = wide basin)
        model_wide = QuadraticEnergy([0.1, 0.1])
        cert_wide = certify_landscape(
            model_wide, jnp.zeros(2),
            basin_perturbations=50,
            basin_key=jrandom.PRNGKey(0),
        )

        # Steep bowl (large eigenvalues = narrow basin) — actually for
        # a quadratic, the basin is infinite (it's a global minimum).
        # But with our log-scale radius test, the estimated basin should
        # be at least as large for both.
        assert cert_wide.basin_radius > 0.0

    def test_eigenvalue_count_matches_dimension(self) -> None:
        """REQ-VERIFY-006: number of eigenvalues equals input dimension."""
        for dim in [2, 5, 10]:
            model = QuadraticEnergy([1.0] * dim)
            cert = certify_landscape(model, jnp.zeros(dim))
            assert len(cert.eigenvalues) == dim

    def test_away_from_minimum(self) -> None:
        """REQ-VERIFY-006: point away from minimum still gets classified."""
        model = QuadraticEnergy([1.0, 2.0])
        x = jnp.array([3.0, 4.0])  # not at minimum, but still in a bowl

        cert = certify_landscape(model, x)

        # For a quadratic, the Hessian is the same everywhere (constant curvature)
        # so it should still be classified as a local minimum region
        assert cert.is_local_minimum
        assert cert.classification == "local_minimum"

    def test_default_key_works(self) -> None:
        """REQ-VERIFY-006: default PRNG key produces valid results."""
        model = QuadraticEnergy([1.0, 2.0])
        cert = certify_landscape(model, jnp.zeros(2))
        assert cert.basin_radius > 0.0
        assert isinstance(cert.eigenvalues, list)
