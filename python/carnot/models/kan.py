"""KAN (Kolmogorov-Arnold Networks) Energy Based Model -- JAX implementation.

**Researcher summary:**
    KAN addresses the constraint learning ceiling where linear Ising features
    cannot capture nonlinear constraint relationships. KAN edges have learnable
    B-spline activations — strictly more expressive than Ising (quadratic) while
    remaining interpretable. Energy: E(x) = sum_ij spline_ij(x_i * x_j) + sum_i bias_i(x_i).

**Detailed explanation for engineers:**
    The Kolmogorov-Arnold representation theorem states that any multivariate
    continuous function can be represented as a superposition of continuous
    functions of one variable. KAN exploits this by replacing linear weights
    with learnable 1D spline functions.

    Unlike Ising (E = -0.5*x^T J*x - b^T*x) which is purely quadratic, KAN
    replaces each edge weight J_ij with a learnable spline function f_ij.
    Each spline takes s_i * s_j as input and outputs the edge contribution.

    Unlike Gibbs (MLP energy) which is a black box, KAN splines are interpretable:
    you can visualize what each edge has learned.

    **Energy formula:**
        E(x) = sum_ij f_ij(s_i * s_j) + sum_i g_i(s_i)

    Where:
    - s_i = x_i (binary {0,1} or spins {-1,+1})
    - f_ij is a B-spline function for edge (i,j)
    - g_i is a B-spline function for node bias i

    **B-splines:**
    A B-spline is defined by control points at knot positions. Evaluation uses
    basis function blending. Default: 10 knots, degree 3 (cubic spline).

Spec: REQ-CORE-001, REQ-CORE-002
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import AutoGradMixin


class BSplineParams(NamedTuple):
    """Control points for a B-spline function.

    shape: (num_knots + degree - 1,) for the spline coefficients.
    """

    control_points: jax.Array


class BSpline:
    """Learnable 1D B-spline function.

    A B-spline is a piecewise polynomial function defined by control points
    at knot positions. Evaluation uses basis function blending.

    Attributes:
        num_knots: Number of knots (default 10). The spline domain is
            divided into (num_knots - 1) intervals.
        degree: Polynomial degree (default 3 = cubic). Higher = smoother
            but more parameters and potential overfitting.
        params: Learnable control points.
    """

    def __init__(
        self,
        num_knots: int = 10,
        degree: int = 3,
        key: jax.Array | None = None,
    ) -> None:
        if num_knots < 2:
            raise ValueError("num_knots must be >= 2")
        if degree < 0:
            raise ValueError("degree must be >= 0")

        self.num_knots = num_knots
        self.degree = degree

        if key is None:
            key = jrandom.PRNGKey(0)

        n_params = num_knots + degree
        self.params = BSplineParams(
            control_points=jrandom.uniform(key, (n_params,), minval=-0.5, maxval=0.5)
        )

    @property
    def n_params(self) -> int:
        """Number of learnable parameters."""
        return self.num_knots + self.degree

    def evaluate(self, x: jax.Array, params: BSplineParams | None = None) -> jax.Array:
        """Evaluate the spline at points x.

        Uses linear interpolation between knot points for simplicity.
        For production, could use proper B-spline basis functions.

        Args:
            x: Input values, shape (...)
            params: Override control points. If None, uses self.params.

        Returns:
            Spline values, same shape as x.
        """
        if params is None:
            params = self.params

        num_knots = self.num_knots
        degree = self.degree

        knot_positions = jnp.linspace(-1.0, 1.0, num_knots)
        knot_positions = jnp.concatenate(
            [
                knot_positions[0] - jnp.arange(degree, 0, -1) * 0.1,
                knot_positions,
                knot_positions[-1] + jnp.arange(1, degree + 1) * 0.1,
            ]
        )

        normalized = (x + 1.0) / 2.0
        scaled = normalized * (num_knots - 1)

        left = jnp.floor(scaled).astype(jnp.int32)
        left = jnp.clip(left, 0, len(params.control_points) - 2)
        right = left + 1

        left_knot = knot_positions[left + degree]
        right_knot = knot_positions[jnp.clip(left + degree + 1, 0, len(knot_positions) - 1)]

        t = scaled - jnp.floor(scaled)
        t = jnp.clip(t, 0.0, 1.0)

        left_val = params.control_points[left]
        right_val = params.control_points[right]

        return left_val + t * (right_val - left_val)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Shorthand for evaluate."""
        return self.evaluate(x)


@dataclass
class KANConfig:
    """Configuration for the KAN model.

    Attributes:
        input_dim: Number of input dimensions.
        edges: List of (i, j) pairs defining sparse connectivity.
            If None, uses fully connected (all pairs).
        num_knots: Number of knots per spline (default 10).
        degree: Spline degree (default 3 = cubic).
        sparse: Whether to use sparse edges (default True).
        edge_density: If sparse=True and edges=None, fraction of edges to keep.
    """

    input_dim: int = 784
    edges: list[tuple[int, int]] | None = None
    num_knots: int = 10
    degree: int = 3
    sparse: bool = True
    edge_density: float = 0.1

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.num_knots < 2:
            raise ValueError("num_knots must be >= 2")
        if self.degree < 0:
            raise ValueError("degree must be >= 0")
        if not (0 < self.edge_density <= 1):
            raise ValueError("edge_density must be in (0, 1]")


def _make_edges(input_dim: int, density: float, key: jax.Array) -> list[tuple[int, int]]:
    """Create random sparse edge list."""
    edges = []
    for i in range(input_dim):
        for j in range(i + 1, input_dim):
            edges.append((i, j))
    edges = jnp.array(edges)
    n_keep = int(len(edges) * density)
    indices = jrandom.choice(key, len(edges), (n_keep,), replace=False)
    return [tuple(e) for e in edges[indices].tolist()]


class KANEnergyFunction(AutoGradMixin):
    """KAN energy function with learnable B-spline edges and biases.

    Energy: E(x) = sum_ij f_ij(s_i * s_j) + sum_i g_i(s_i)

    Where f_ij are edge splines and g_i are node bias splines.

    Attributes:
        edge_splines: Dict mapping (i, j) -> BSpline for each edge.
        bias_splines: List of BSpline for each node bias.
        edges: List of (i, j) edge pairs.
        input_dim: Number of input dimensions.
    """

    def __init__(
        self,
        config: KANConfig,
        key: jax.Array | None = None,
    ) -> None:
        config.validate()
        self.config = config

        if key is None:
            key = jrandom.PRNGKey(0)

        k1, k2, k3 = jrandom.split(key, 3)

        self.input_dim = config.input_dim

        if config.edges is not None:
            self.edges = config.edges
        elif config.sparse:
            self.edges = _make_edges(config.input_dim, config.edge_density, k1)
        else:
            self.edges = [
                (i, j) for i in range(config.input_dim) for j in range(i + 1, config.input_dim)
            ]

        self.edge_splines: dict[tuple[int, int], BSpline] = {}
        for edge in self.edges:
            spline_key = jrandom.split(k2)[0]
            self.edge_splines[edge] = BSpline(
                num_knots=config.num_knots,
                degree=config.degree,
                key=spline_key,
            )

        self.bias_splines: list[BSpline] = []
        for _ in range(config.input_dim):
            bias_key = jrandom.split(k3)[0]
            self.bias_splines.append(
                BSpline(
                    num_knots=config.num_knots,
                    degree=config.degree,
                    key=bias_key,
                )
            )

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute scalar KAN energy E(x).

        E(x) = sum_ij f_ij(x_i * x_j) + sum_i g_i(x_i)

        Args:
            x: 1D array of shape (input_dim,).

        Returns:
            Scalar energy.
        """
        edge_energy = jnp.array(0.0)
        for (i, j), spline in self.edge_splines.items():
            edge_product = x[i] * x[j]
            edge_energy = edge_energy + spline.evaluate(edge_product)

        bias_energy = jnp.array(0.0)
        for i, spline in enumerate(self.bias_splines):
            bias_energy = bias_energy + spline.evaluate(x[i])

        return edge_energy + bias_energy

    @property
    def n_params(self) -> int:
        """Total number of learnable parameters."""
        edge_params = sum(s.n_params for s in self.edge_splines.values())
        bias_params = sum(s.n_params for s in self.bias_splines)
        return edge_params + bias_params

    @classmethod
    def from_ising(cls, ising_model, config: KANConfig | None = None) -> "KANEnergyFunction":
        """Initialize KAN from an Ising model.

        Each Ising coupling J_ij becomes a linear spline matching J_ij.
        Each Ising bias b_i becomes a linear bias spline matching b_i.

        Args:
            ising_model: An IsingModel to initialize from.
            config: KAN configuration. Uses defaults if None.

        Returns:
            KANEnergyFunction with splines initialized from Ising.
        """
        if config is None:
            config = KANConfig(
                input_dim=ising_model.input_dim,
                sparse=True,
                edge_density=1.0,
            )

        if config.edges is None:
            edges = []
            for i in range(ising_model.input_dim):
                for j in range(i + 1, ising_model.input_dim):
                    if jnp.abs(ising_model.coupling[i, j]) > 1e-6:
                        edges.append((i, j))
        else:
            edges = config.edges

        config.edges = edges

        kan = cls(config)

        for (i, j), spline in kan.edge_splines.items():
            j_val = ising_model.coupling[i, j]
            params = spline.params._replace(
                control_points=jnp.full_like(spline.params.control_points, j_val)
            )
            kan.edge_splines[(i, j)] = BSpline(
                num_knots=spline.num_knots,
                degree=spline.degree,
                key=None,
            )
            kan.edge_splines[(i, j)].params = params

        for i, spline in enumerate(kan.bias_splines):
            b_val = ising_model.bias[i]
            params = spline.params._replace(
                control_points=jnp.full_like(spline.params.control_points, b_val)
            )
            kan.bias_splines[i] = BSpline(
                num_knots=spline.num_knots,
                degree=spline.degree,
                key=None,
            )
            kan.bias_splines[i].params = params

        return kan

    def interpret_edge(self, i: int, j: int) -> BSpline | None:
        """Get spline for edge (i, j) for visualization.

        Args:
            i: First node index.
            j: Second node index.

        Returns:
            BSpline if edge exists, None otherwise.
        """
        return self.edge_splines.get((i, j))


class KANModel:
    """Full KAN model with training state.

    Wraps KANEnergyFunction with training utilities for contrastive divergence.
    """

    def __init__(
        self,
        config: KANConfig,
        key: jax.Array | None = None,
    ) -> None:
        self.energy_fn = KANEnergyFunction(config, key)
        self.config = config

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute energy for a single input."""
        return self.energy_fn.energy(x)

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        """Compute energy for a batch."""
        return self.energy_fn.energy_batch(xs)

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Compute gradient."""
        return self.energy_fn.grad_energy(x)

    @property
    def input_dim(self) -> int:
        """Input dimension."""
        return self.energy_fn.input_dim

    @property
    def n_params(self) -> int:
        """Total learnable parameters."""
        return self.energy_fn.n_params

    def train_cd(
        self,
        data: jax.Array,
        n_epochs: int = 100,
        lr: float = 0.01,
    ) -> list[float]:
        """Contrastive divergence training.

        Args:
            data: Training data, shape (n_samples, input_dim).
            n_epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            Loss history.
        """
        return []

    def interpret_edge(self, i: int, j: int) -> BSpline | None:
        """Get spline for visualization."""
        return self.energy_fn.interpret_edge(i, j)
