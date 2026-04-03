"""Analytical benchmark energy functions — JAX implementation.

**Researcher summary:**
    Five classical test functions (DoubleWell, Rosenbrock, Ackley, Rastrigin,
    GaussianMixture) implementing EnergyFunction, with known global minima.
    Gradients via jax.grad (AutoGradMixin). Matches Rust implementations in
    crates/carnot-core/src/benchmarks/mod.rs.

**Detailed explanation for engineers:**
    When developing MCMC samplers or training algorithms, you need test
    functions where you KNOW the correct answer. These benchmarks provide
    exactly that — simple mathematical functions with known minima, so you
    can measure how well your algorithm is actually working.

    All benchmarks inherit from AutoGradMixin, which provides:
    - grad_energy via jax.grad (exact, not numerical)
    - energy_batch via jax.vmap (vectorized over batch dim)

    You only need to implement energy(x) for a single input.

Spec: REQ-AUTO-001
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from carnot.core.energy import AutoGradMixin


@dataclass
class BenchmarkInfo:
    """Metadata about a benchmark's known optimal solution.

    **For engineers:**
        Every benchmark comes with a "cheat sheet" — this tells you exactly
        where the minimum is and what energy it has. Compare your sampler's
        output against this to measure performance.

    Attributes:
        name: Machine-readable identifier (e.g., "double_well").
        input_dim: Dimensionality of the input space.
        global_min_energy: Energy at the global minimum (usually 0.0).
        global_min_location: Exact location of the global minimum.
        description: Human-readable formula and properties.
    """

    name: str
    input_dim: int
    global_min_energy: float
    global_min_location: jax.Array
    description: str


class DoubleWell(AutoGradMixin):
    """Double-well potential: tests multimodal sampling.

    **Researcher summary:**
        Quartic double-well in x[0] with quadratic confinement in remaining
        dims. E(x) = (x[0]^2 - 1)^2 + sum(x[1:]^2). Two symmetric minima
        at x[0] = +/-1 with E = 0. Barrier height = 1.0 at x[0] = 0.

    **Detailed explanation for engineers:**
        Imagine a landscape shaped like a "W" — two valleys at x[0] = -1
        and x[0] = +1 (both with energy = 0), separated by a hill at
        x[0] = 0 (energy = 1). The remaining dimensions are simple
        quadratic bowls centered at zero.

        This tests the hardest problem in MCMC: can the sampler jump
        between the two valleys? A naive sampler gets stuck in one valley.
        Good samplers (especially HMC with enough momentum) cross the barrier.

    Spec: REQ-AUTO-001
    """

    def __init__(self, dim: int = 2) -> None:
        if dim < 1:
            raise ValueError("DoubleWell requires dim >= 1")
        self._dim = dim

    def energy(self, x: jax.Array) -> jax.Array:
        """E(x) = (x[0]^2 - 1)^2 + sum(x[1:]^2)."""
        well = (x[0] ** 2 - 1.0) ** 2
        rest = jnp.sum(x[1:] ** 2)
        return well + rest

    @property
    def input_dim(self) -> int:
        return self._dim

    def info(self) -> BenchmarkInfo:
        loc = jnp.zeros(self._dim).at[0].set(1.0)
        return BenchmarkInfo(
            name="double_well",
            input_dim=self._dim,
            global_min_energy=0.0,
            global_min_location=loc,
            description="E(x) = (x[0]^2 - 1)^2 + sum(x[1:]^2). Two minima at x[0]=+/-1.",
        )


class Rosenbrock(AutoGradMixin):
    """Rosenbrock function: tests navigation of narrow curved valleys.

    **Researcher summary:**
        E(x) = sum_{i=0}^{n-2} [100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2].
        Global minimum at x = [1, 1, ..., 1] with E = 0. The narrow, curved
        valley ("banana" shape) makes this highly ill-conditioned.

    **Detailed explanation for engineers:**
        The Rosenbrock "banana" function has a long, narrow, curved valley.
        Finding the valley is easy — most starting points descend into it
        quickly. But the minimum WITHIN the valley is hard to reach because
        the floor slopes very gently while the walls are steep.

        This tests whether your sampler handles highly correlated dimensions
        (x[i+1] wants to be close to x[i]^2).

    Spec: REQ-AUTO-001
    """

    def __init__(self, dim: int = 2) -> None:
        if dim < 2:
            raise ValueError("Rosenbrock requires dim >= 2")
        self._dim = dim

    def energy(self, x: jax.Array) -> jax.Array:
        """E(x) = sum [100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2]."""
        a = x[1:] - x[:-1] ** 2  # x[i+1] - x[i]^2
        b = 1.0 - x[:-1]  # 1 - x[i]
        return jnp.sum(100.0 * a**2 + b**2)

    @property
    def input_dim(self) -> int:
        return self._dim

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="rosenbrock",
            input_dim=self._dim,
            global_min_energy=0.0,
            global_min_location=jnp.ones(self._dim),
            description="Rosenbrock banana function. Min at [1,1,...,1].",
        )


class Ackley(AutoGradMixin):
    """Ackley function: tests escape from many local minima.

    **Researcher summary:**
        E(x) = -20*exp(-0.2*sqrt(mean(x^2))) - exp(mean(cos(2*pi*x))) + 20 + e.
        Global minimum at origin with E = 0. Dense grid of local minima from
        the cosine term traps gradient-based methods.

    **Detailed explanation for engineers:**
        The Ackley function looks like a bumpy surface with one deep hole at
        the origin. Away from the origin, it's nearly flat but covered with
        small bumps (local minima from the cosine terms). Like a golf course
        with many sand traps — the hole is at the center but a ball keeps
        getting caught in traps.

    Spec: REQ-AUTO-001
    """

    def __init__(self, dim: int = 2) -> None:
        if dim < 1:
            raise ValueError("Ackley requires dim >= 1")
        self._dim = dim

    def energy(self, x: jax.Array) -> jax.Array:
        """E(x) = -20*exp(-0.2*sqrt(mean(x^2))) - exp(mean(cos(2*pi*x))) + 20 + e.

        Note: uses sqrt(mean(x^2) + 1e-10) to avoid NaN gradient at origin,
        since d/dx sqrt(x) is undefined at x=0.
        """
        n = x.shape[0]
        sum_sq = jnp.sum(x**2)
        sum_cos = jnp.sum(jnp.cos(2.0 * jnp.pi * x))
        # Small epsilon inside sqrt to keep gradient finite at origin
        return (
            -20.0 * jnp.exp(-0.2 * jnp.sqrt(sum_sq / n + 1e-10))
            - jnp.exp(sum_cos / n)
            + 20.0
            + jnp.e
        )

    @property
    def input_dim(self) -> int:
        return self._dim

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="ackley",
            input_dim=self._dim,
            global_min_energy=0.0,
            global_min_location=jnp.zeros(self._dim),
            description="Ackley multimodal function. Min at origin.",
        )


class Rastrigin(AutoGradMixin):
    """Rastrigin function: tests systematic local minima avoidance.

    **Researcher summary:**
        E(x) = 10*n + sum(x[i]^2 - 10*cos(2*pi*x[i])). Global minimum at
        origin with E = 0. ~10^n local minima at integer lattice points.
        Cosine modulation amplitude 10 creates deep basins.

    **Detailed explanation for engineers:**
        Like a waffle iron — a regular grid of equally-spaced bumps with the
        deepest point at the origin. Unlike Ackley (one deep hole + shallow
        bumps), Rastrigin has deep local minima at EVERY integer point. In
        2D there are hundreds; in 10D there are 10 billion.

    Spec: REQ-AUTO-001
    """

    def __init__(self, dim: int = 2) -> None:
        if dim < 1:
            raise ValueError("Rastrigin requires dim >= 1")
        self._dim = dim

    def energy(self, x: jax.Array) -> jax.Array:
        """E(x) = 10*n + sum(x[i]^2 - 10*cos(2*pi*x[i]))."""
        n = x.shape[0]
        return 10.0 * n + jnp.sum(x**2 - 10.0 * jnp.cos(2.0 * jnp.pi * x))

    @property
    def input_dim(self) -> int:
        return self._dim

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="rastrigin",
            input_dim=self._dim,
            global_min_energy=0.0,
            global_min_location=jnp.zeros(self._dim),
            description="Rastrigin function with many regular local minima. Min at origin.",
        )


class GaussianMixture(AutoGradMixin):
    """Gaussian mixture energy: gold-standard benchmark with known distribution.

    **Researcher summary:**
        E(x) = -log(sum_k w_k * N(x; mu_k, sigma_k^2 * I)). Log-sum-exp
        for numerical stability. Known parameters allow exact KL divergence,
        mode coverage, and distributional metrics.

    **Detailed explanation for engineers:**
        This is the most important benchmark because we know the EXACT target
        distribution, not just the minimum. A Gaussian mixture is a sum of
        bell curves centered at different locations. The energy is the negative
        log-probability: low energy = high probability.

        A correct sampler should produce samples proportional to each mode's
        weight. If it produces 90% near one mode and 10% near the other when
        weights are equal, it has a mode imbalance problem.

    Spec: REQ-AUTO-001
    """

    def __init__(
        self,
        dim: int,
        means: list[jax.Array],
        variances: list[float],
        weights: list[float],
    ) -> None:
        if dim < 1:
            raise ValueError("GaussianMixture requires dim >= 1")
        if len(means) != len(variances) or len(means) != len(weights):
            raise ValueError("means, variances, and weights must have equal length")
        self._dim = dim
        self.means = means
        self.variances = jnp.array(variances)
        self.weights = jnp.array(weights)

    @classmethod
    def two_modes(cls, separation: float = 4.0) -> GaussianMixture:
        """Create a 1D mixture of two equal-weight Gaussians.

        Modes at -separation/2 and +separation/2, unit variance, equal weight.
        Larger separation makes mixing harder.
        """
        return cls(
            dim=1,
            means=[
                jnp.array([-separation / 2.0]),
                jnp.array([separation / 2.0]),
            ],
            variances=[1.0, 1.0],
            weights=[0.5, 0.5],
        )

    def energy(self, x: jax.Array) -> jax.Array:
        """E(x) = -log(sum_k w_k * N(x; mu_k, sigma_k^2 * I)).

        Uses log-sum-exp for numerical stability.
        """
        log_components = []
        for i, mean in enumerate(self.means):
            diff = x - mean
            exponent = -0.5 * jnp.sum(diff**2) / self.variances[i]
            log_norm = -0.5 * self._dim * jnp.log(2.0 * jnp.pi * self.variances[i])
            log_w = jnp.log(self.weights[i])
            log_components.append(log_w + exponent + log_norm)

        log_sum = jax.nn.logsumexp(jnp.array(log_components))
        return -log_sum

    @property
    def input_dim(self) -> int:
        return self._dim

    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="gaussian_mixture",
            input_dim=self._dim,
            global_min_energy=0.0,
            global_min_location=self.means[0],
            description="Gaussian mixture model with known parameters.",
        )


def get_standard_benchmarks(dim: int = 2) -> dict[str, Any]:
    """Get all standard benchmarks at the given dimensionality.

    Returns a dict mapping benchmark name to (energy_function, info) pairs.
    This is the primary entry point for the autoresearch pipeline.

    Args:
        dim: Input dimensionality for all benchmarks. Default 2.

    Returns:
        Dict of {"name": (EnergyFunction, BenchmarkInfo)} pairs.
    """
    benchmarks = {
        "double_well": DoubleWell(dim),
        "rosenbrock": Rosenbrock(max(dim, 2)),
        "ackley": Ackley(dim),
        "rastrigin": Rastrigin(dim),
        "gaussian_mixture": GaussianMixture.two_modes(4.0) if dim == 1
        else GaussianMixture(
            dim=dim,
            means=[jnp.full(dim, -2.0), jnp.full(dim, 2.0)],
            variances=[1.0, 1.0],
            weights=[0.5, 0.5],
        ),
    }
    return {
        name: (fn, fn.info())  # type: ignore[attr-defined]
        for name, fn in benchmarks.items()
    }
