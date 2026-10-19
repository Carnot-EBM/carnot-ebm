"""Graph coloring constraint satisfaction -- JAX implementation.

**Researcher summary:**
    Encodes graph coloring as differentiable energy terms. Each node has a
    continuous color value; adjacent nodes must differ. Uses pairwise repulsion
    (identical to Sudoku uniqueness) plus a range penalty.

**Detailed explanation for engineers:**
    Graph coloring assigns a color to each node such that no two adjacent
    nodes share a color. This module encodes it as an EBM:

    1. **Color difference constraints** (one per edge): For edge (a, b),
       E = max(0, 1 - |x_a - x_b|)^2. This is the same pairwise repulsion
       used in Sudoku's UniquenessConstraint.

    2. **Range penalty** (one total): Keeps colors in [0, n_colors - 1].
       E = sum(max(0, -x_i)^2 + max(0, x_i - (n_colors - 1))^2).

    **Why is this even simpler than SAT?** Each constraint involves exactly
    two variables (one edge), gradient flows are direct, and the energy
    landscape is relatively smooth. Good for verifying the LLM-EBM pipeline
    works before tackling harder domains.

Spec: REQ-INFER-002, SCENARIO-INFER-003
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from carnot.verify.constraint import BaseConstraint, ComposedEnergy


class ColorDifferenceConstraint(BaseConstraint):
    """Two adjacent nodes must have different colors.

    **Researcher summary:**
        E = max(0, 1 - |x_a - x_b|)^2. Zero when colors differ by >= 1.
        Identical formulation to Sudoku's pairwise repulsion.

    **Detailed explanation for engineers:**
        For an edge between node_a and node_b, this checks that their
        color values are sufficiently different. The energy is:

        - |x_a - x_b| >= 1: energy = 0 (different colors)
        - |x_a - x_b| < 1: energy = (1 - |x_a - x_b|)^2 (penalty)

        Maximum penalty (1.0) when x_a == x_b (same color).

    For example::

        constraint = ColorDifferenceConstraint("edge_0_1", 0, 1)
        x = jnp.array([0.0, 1.0, 2.0])  # nodes 0,1,2 all different
        assert constraint.is_satisfied(x)  # True

    Spec: REQ-INFER-002, SCENARIO-INFER-003
    """

    def __init__(self, name: str, node_a: int, node_b: int) -> None:
        self._name = name
        self._node_a = node_a
        self._node_b = node_b

    @property
    def name(self) -> str:
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        return 0.01

    def energy(self, x: jax.Array) -> jax.Array:
        """Pairwise repulsion energy for this edge.

        Spec: REQ-INFER-002
        """
        diff = jnp.abs(x[self._node_a] - x[self._node_b])
        violation = jnp.maximum(1.0 - diff, 0.0)
        return violation**2


class ColorRangeConstraint(BaseConstraint):
    """Soft penalty keeping colors in valid range [0, n_colors - 1].

    **Researcher summary:**
        Boundary penalty: E = sum(max(0, -x_i)^2 + max(0, x_i - max_color)^2).
        Zero when all colors are in range.

    **Detailed explanation for engineers:**
        After gradient repair, color values might drift outside [0, n_colors-1].
        This penalty pushes them back into range:

        - x_i < 0: penalty = x_i^2
        - x_i > n_colors - 1: penalty = (x_i - (n_colors - 1))^2
        - Otherwise: penalty = 0

    Spec: REQ-INFER-002
    """

    def __init__(self, name: str, node_indices: list[int], n_colors: int) -> None:
        self._name = name
        self._node_indices = node_indices
        self._n_colors = n_colors

    @property
    def name(self) -> str:
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        return 0.01

    def energy(self, x: jax.Array) -> jax.Array:
        """Boundary penalty energy.

        Spec: REQ-INFER-002
        """
        vals = x[jnp.array(self._node_indices)]
        max_color = float(self._n_colors - 1)
        below = jnp.sum(jnp.maximum(-vals, 0.0) ** 2)
        above = jnp.sum(jnp.maximum(vals - max_color, 0.0) ** 2)
        return below + above


def build_coloring_energy(
    edges: list[tuple[int, int]],
    n_nodes: int,
    n_colors: int,
    edge_weight: float = 1.0,
    range_weight: float = 0.5,
) -> ComposedEnergy:
    """Build ComposedEnergy for a graph coloring instance.

    **Researcher summary:**
        One constraint per edge (weight 1.0) plus one range penalty
        (weight 0.5). Returns ComposedEnergy ready for verify/repair.

    **Detailed explanation for engineers:**
        Mirrors ``build_sudoku_energy`` and ``build_sat_energy``: creates
        a ComposedEnergy with input_dim=n_nodes, adds one
        ColorDifferenceConstraint per edge and one ColorRangeConstraint
        for all nodes.

    Args:
        edges: List of (node_a, node_b) tuples defining the graph.
        n_nodes: Number of nodes in the graph.
        n_colors: Number of available colors.
        edge_weight: Weight for each edge constraint.
        range_weight: Weight for the range penalty.

    Returns:
        ComposedEnergy with input_dim=n_nodes.

    Spec: REQ-INFER-002
    """
    composed = ComposedEnergy(input_dim=n_nodes)

    for _i, (node_a, node_b) in enumerate(edges):
        composed.add_constraint(
            ColorDifferenceConstraint(f"edge_{node_a}_{node_b}", node_a, node_b),
            edge_weight,
        )

    composed.add_constraint(
        ColorRangeConstraint("color_range", list(range(n_nodes)), n_colors),
        range_weight,
    )

    return composed


def coloring_to_array(colors: list[int]) -> jax.Array:
    """Convert integer color list to JAX float array.

    Spec: REQ-INFER-002
    """
    return jnp.array([float(c) for c in colors])


def array_to_coloring(x: jax.Array) -> list[int]:
    """Round continuous array to integer colors.

    Spec: REQ-INFER-002
    """
    return [int(round(float(v))) for v in x]
