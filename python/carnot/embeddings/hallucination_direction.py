"""Hallucination direction detection via activation-space analysis.

**Researcher summary:**
    Given per-layer activations from correct and hallucinated LLM outputs,
    finds the principal direction in activation space that distinguishes them.
    This direction becomes an energy function: high projection onto the
    hallucination direction = high energy = likely hallucination.

**Detailed explanation for engineers:**
    Large language models produce hidden-state activations at each transformer
    layer during inference. When the model generates factually correct text,
    these activations cluster in one region of the high-dimensional space.
    When it hallucinates (generates plausible-sounding but incorrect text),
    the activations shift systematically in a particular direction.

    This module discovers that direction by:

    1. **Mean difference**: Compute the mean activation vector for correct
       outputs and for hallucinated outputs, then take their difference.
       This "hallucination direction" vector points from the correct cluster
       toward the hallucination cluster.

    2. **SVD refinement** (optional): Instead of a single mean-difference
       vector, use Singular Value Decomposition to find the top-k directions
       that best separate the two distributions. This captures cases where
       hallucination manifests along multiple correlated dimensions.

    3. **Energy function**: The projection of any new activation onto the
       hallucination direction gives a scalar "hallucination energy." High
       energy = the activation looks more like known hallucinations. This
       energy can be used as a real-time signal during generation, or
       composed with other Carnot constraints via ``ComposedEnergy``.

    **Integration with Carnot's constraint system:**
    ``HallucinationDirectionConstraint`` wraps this energy as a
    ``BaseConstraint``, so it can be added to a ``ComposedEnergy`` alongside
    other constraints (e.g., factual grounding, coherence). The repair
    pipeline can then gradient-descend on the hallucination energy to push
    activations away from the hallucination direction.

Spec: REQ-INFER-014
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from carnot.verify.constraint import BaseConstraint


@dataclass
class HallucinationDirectionConfig:
    """Configuration for hallucination direction discovery.

    **Detailed explanation for engineers:**
        Controls how the hallucination direction is computed from activation
        data:

        - ``top_k``: When > 1, uses SVD to find multiple distinguishing
          directions instead of just the mean difference. The returned
          direction matrix has shape ``(top_k, hidden_dim)``. Set to 1
          (default) for the simple mean-difference approach.

        - ``normalize``: Whether to L2-normalize the resulting direction
          vector(s). Normalization makes the energy function output
          interpretable as a cosine-like projection (range roughly -1 to +1
          per direction). Without normalization, the magnitude of the
          direction encodes how far apart the clusters are, which may be
          useful for weighting.

    Attributes:
        top_k: Number of principal directions to extract via SVD.
            Must be >= 1. Default 1 (single mean-difference vector).
        normalize: Whether to L2-normalize direction vectors. Default True.

    Spec: REQ-INFER-014
    """

    top_k: int = 1
    normalize: bool = True

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If top_k < 1.
        """
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")


def find_hallucination_direction(
    correct_activations: list[jax.Array],
    hallucinated_activations: list[jax.Array],
    config: Optional[HallucinationDirectionConfig] = None,
) -> jax.Array:
    """Find the principal direction(s) separating correct from hallucinated activations.

    **Researcher summary:**
        Computes mean_hallucinated - mean_correct as the hallucination
        direction. Optionally uses SVD on the centered difference matrix
        to extract top-k distinguishing directions.

    **Detailed explanation for engineers:**
        The algorithm works in two modes depending on ``config.top_k``:

        **Mode 1: Mean difference (top_k=1, default)**
        Simply computes ``mean(hallucinated) - mean(correct)``. This single
        vector points from the "correct" centroid toward the "hallucinated"
        centroid in activation space. Fast, interpretable, and works well
        when hallucinations shift activations along one dominant direction.

        **Mode 2: SVD (top_k > 1)**
        Stacks all activations into a matrix, labels them +1 (hallucinated)
        or -1 (correct), centers them, and runs SVD. The top-k right
        singular vectors capture the directions of maximum variance between
        the two groups. This is essentially a form of Linear Discriminant
        Analysis (LDA) simplified for the two-class case.

        In both modes, the result is optionally L2-normalized so that
        projections onto the direction(s) have interpretable magnitudes.

    Args:
        correct_activations: List of activation vectors from correct outputs.
            Each element is a JAX array of shape ``(hidden_dim,)``.
            Must contain at least one element.
        hallucinated_activations: List of activation vectors from hallucinated
            outputs. Same shape requirements as correct_activations.
            Must contain at least one element.
        config: Configuration controlling top_k and normalization.
            Uses defaults if None.

    Returns:
        If top_k == 1: a JAX array of shape ``(hidden_dim,)`` — the single
        hallucination direction vector.
        If top_k > 1: a JAX array of shape ``(top_k, hidden_dim)`` — the
        top-k hallucination direction vectors as rows.

    Raises:
        ValueError: If either activation list is empty, or if activations
            have mismatched dimensions.

    Spec: REQ-INFER-014
    """
    if config is None:
        config = HallucinationDirectionConfig()
    config.validate()

    if len(correct_activations) == 0:
        raise ValueError("correct_activations must not be empty")
    if len(hallucinated_activations) == 0:
        raise ValueError("hallucinated_activations must not be empty")

    # Stack into matrices: (n_correct, hidden_dim) and (n_hallucinated, hidden_dim)
    correct_mat = jnp.stack(correct_activations)
    halluc_mat = jnp.stack(hallucinated_activations)

    if correct_mat.shape[1:] != halluc_mat.shape[1:]:
        raise ValueError(
            f"Activation dimensions must match: correct has shape {correct_mat.shape[1:]}, "
            f"hallucinated has shape {halluc_mat.shape[1:]}"
        )

    # Compute centroids.
    mean_correct = jnp.mean(correct_mat, axis=0)
    mean_halluc = jnp.mean(halluc_mat, axis=0)

    if config.top_k == 1:
        # Simple mean-difference direction.
        direction = mean_halluc - mean_correct
        if config.normalize:
            norm = jnp.linalg.norm(direction)
            direction = direction / jnp.maximum(norm, 1e-8)
        return direction

    # SVD mode: find top-k directions of maximum separation.
    # Center all activations around the global mean, then weight by class label
    # so that the SVD captures between-class variance.
    global_mean = (mean_correct + mean_halluc) / 2.0
    centered_correct = correct_mat - global_mean
    centered_halluc = halluc_mat - global_mean

    # Stack with sign: hallucinated gets +1, correct gets -1.
    # This ensures SVD's top singular vectors align with the separation axis.
    signed = jnp.concatenate([-centered_correct, centered_halluc], axis=0)

    # Thin SVD: we only need the top-k right singular vectors.
    _u, _s, vt = jnp.linalg.svd(signed, full_matrices=False)
    directions = vt[:config.top_k]

    if config.normalize:
        # Each row is already unit-norm from SVD, but normalize explicitly
        # for safety in case of numerical drift.
        norms = jnp.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / jnp.maximum(norms, 1e-8)

    return directions


def hallucination_energy(
    activation: jax.Array,
    direction: jax.Array,
) -> jax.Array:
    """Compute hallucination energy: projection of activation onto direction.

    **Researcher summary:**
        Energy = dot(activation, direction) / ||direction||. High projection
        onto the hallucination direction means the activation resembles known
        hallucinations.

    **Detailed explanation for engineers:**
        This is a simple linear energy function. Given an activation vector
        and a hallucination direction (from ``find_hallucination_direction``),
        the energy is the signed scalar projection of the activation onto
        that direction.

        - **Positive energy**: the activation has a component pointing in the
          hallucination direction — it looks more like a hallucination.
        - **Negative energy**: the activation points away from the
          hallucination direction — it looks more like a correct output.
        - **Zero energy**: the activation is orthogonal to the hallucination
          direction — no signal either way.

        If the direction is already normalized (which it is by default from
        ``find_hallucination_direction``), this simplifies to a dot product.

        For multi-direction mode (direction shape ``(k, hidden_dim)``), the
        energy is the sum of squared projections onto each direction, which
        captures the total variance along the hallucination subspace.

    Args:
        activation: A single activation vector, shape ``(hidden_dim,)``.
        direction: The hallucination direction from
            ``find_hallucination_direction``. Shape ``(hidden_dim,)`` for
            single direction, or ``(k, hidden_dim)`` for multi-direction.

    Returns:
        Scalar energy value (JAX array with shape ``()``).

    Spec: REQ-INFER-014
    """
    if direction.ndim == 1:
        # Single direction: signed projection.
        norm = jnp.linalg.norm(direction)
        return jnp.dot(activation, direction) / jnp.maximum(norm, 1e-8)
    else:
        # Multiple directions: sum of squared projections onto each.
        # This gives the total energy in the hallucination subspace.
        norms = jnp.linalg.norm(direction, axis=1, keepdims=True)
        normed_dirs = direction / jnp.maximum(norms, 1e-8)
        projections = normed_dirs @ activation  # shape (k,)
        return jnp.sum(projections**2)


class HallucinationDirectionConstraint(BaseConstraint):
    """Constraint that penalizes activations projecting onto the hallucination direction.

    **Researcher summary:**
        Wraps ``hallucination_energy`` as a ``BaseConstraint`` for use with
        ``ComposedEnergy``. Energy is high when activation aligns with the
        hallucination direction; the repair pipeline can then push it away.

    **Detailed explanation for engineers:**
        This class bridges the hallucination direction analysis with Carnot's
        constraint-based verification system. By expressing "don't hallucinate"
        as a differentiable energy constraint, we can:

        1. **Compose** it with other constraints (factual grounding, coherence,
           format compliance) in a ``ComposedEnergy``.
        2. **Verify** whether a model output's activations are in the
           "hallucination zone" using ``ComposedEnergy.verify()``.
        3. **Repair** hallucinating activations by gradient descent using
           ``repair()`` — literally pushing the hidden states away from the
           hallucination direction.

        The energy is computed as ``max(0, projection)`` — we only penalize
        positive projections (toward hallucination), not negative ones (away
        from hallucination). This is a ReLU-style one-sided penalty that
        lets the constraint be fully satisfied (energy = 0) when the
        activation doesn't align with hallucination.

        The ``satisfaction_threshold`` is configurable (default 0.1) because
        hallucination detection is inherently soft — we don't expect exact
        zero projection.

    Attributes:
        direction: The hallucination direction vector(s).
        threshold: Energy threshold below which the constraint is satisfied.

    Spec: REQ-INFER-014
    """

    def __init__(
        self,
        direction: jax.Array,
        threshold: float = 0.1,
        constraint_name: str = "hallucination_direction",
    ) -> None:
        """Initialize the hallucination direction constraint.

        Args:
            direction: Hallucination direction from
                ``find_hallucination_direction``. Shape ``(hidden_dim,)``
                or ``(k, hidden_dim)``.
            threshold: Satisfaction threshold. Activations with energy below
                this are considered "not hallucinating." Default 0.1 (soft).
            constraint_name: Human-readable name for reports. Default
                ``"hallucination_direction"``.
        """
        self._direction = direction
        self._threshold = threshold
        self._name = constraint_name

    @property
    def name(self) -> str:
        """Human-readable constraint name for verification reports."""
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        """Threshold below which this constraint is considered satisfied.

        Hallucination detection is soft — we use a higher threshold (0.1)
        than the default BaseConstraint (1e-6) because zero projection
        onto the hallucination direction is unlikely in practice.
        """
        return self._threshold

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute one-sided hallucination energy for activation x.

        **Detailed explanation for engineers:**
            Computes the projection of x onto the hallucination direction,
            then applies ReLU (max with 0) so that only positive projections
            (toward hallucination) contribute energy. Negative projections
            (away from hallucination) give zero energy — they're fine.

            This makes the constraint one-sided: it penalizes hallucination
            but doesn't reward being "maximally anti-hallucination."

        Args:
            x: Activation vector, shape ``(hidden_dim,)``.

        Returns:
            Scalar non-negative energy. Zero if x doesn't project onto
            the hallucination direction.

        Spec: REQ-INFER-014
        """
        raw_energy = hallucination_energy(x, self._direction)
        # One-sided: only penalize positive projection (toward hallucination).
        return jnp.maximum(raw_energy, 0.0)
