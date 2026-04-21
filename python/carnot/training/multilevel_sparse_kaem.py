"""MultilevelSparseKAEMTrainer — Multilevel training for SparseKAEMEnergy.

**Why combine multilevel and sparse (the RETRO-057 carry-5 hypothesis):**
    RETRO-057 carry 5 identified two partial solutions:
    - MultilevelKAEMTrainer (Exp 634): coarse-to-fine knot refinement avoids local
      minima by starting on a smooth landscape and refining progressively.
    - SparseKAEMEnergy (Exp 637): sparse pairwise coupling drops the weakest
      interaction terms while keeping full per-variable univariate splines.

    Both approaches alone failed to reach the 5% energy accuracy threshold vs dense:
    - Multilevel alone: accuracy_multilevel = 8.49 (worse than standard 2.91)
    - Sparse alone: sparse_vs_dense_error = 0.429 (far outside 5% = 0.05 target)

    This module combines them: train SparseKAEMEnergy at coarse resolution (K=16),
    then progressively refine knot count (K=32, K=64) while re-sparsifying after each
    level.  The hypothesis is that:
    1. Coarse-to-fine avoids the poor local minima that plagued sparse-only training.
    2. Sparsification at each level prevents the fine-resolution model from overfitting
       to the noisy coupling structure learned at coarse resolution.
    3. Together they may produce energy accuracy within the 5% tolerance.

**Mathematical intuition:**
    At each level the model has N_knots * n_vars univariate parameters + a sparse
    coupling matrix with at most top_k * n_vars non-zero entries.  Coarse levels have
    fewer knot parameters so the gradient landscape is smoother — easier for SGD to
    find the basin of the global minimum.  As we refine, the already-learned coupling
    structure provides a warm start for the denser coupling matrix.

Spec: REQ-SAMPLE-025, SCENARIO-SAMPLE-040, SCENARIO-SAMPLE-041
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from carnot.models.sparse_kaem_energy import SparseKAEMEnergy
from carnot.training.multilevel_kan_trainer import KnotRefinementInterpolator


# ---------------------------------------------------------------------------
# MultilevelSparseKAEMTrainer
# ---------------------------------------------------------------------------


class MultilevelSparseKAEMTrainer:
    """Train SparseKAEMEnergy using a multilevel knot refinement schedule.

    Each training level:
    1. (First level only) Initialises SparseKAEMEnergy at schedule[0] knots.
    2. (Subsequent levels) Interpolates univariate spline control points from the
       previous level to the next finer knot count using KnotRefinementInterpolator.
    3. Runs epochs_per_level epochs of SparseKAEMEnergy.fit() (score matching +
       coupling gradient descent).
    4. Re-applies sparsification: zeroes out below-threshold coupling entries.

    The coupling matrix is NOT interpolated between levels — only the per-variable
    univariate spline control points are.  The coupling matrix is re-initialised
    (small random) at each level transition.  This is intentional: the coupling
    matrix is typically much smaller than the spline parameters, and re-initialising
    it avoids carrying over stale coupling structure from the coarse level.

    Parameters
    ----------
    schedule : list[int]
        Knot counts per level in strictly increasing order.
        Default [16, 32, 64].
    epochs_per_level : int
        Number of training epochs at each resolution level.
        Default 20.
    top_k_fraction : float
        Fraction of n_vars to retain as active couplings per variable.
        Passed directly to SparseKAEMEnergy.  Default 0.1.

    Spec: REQ-SAMPLE-025
    """

    def __init__(
        self,
        schedule: list[int] | None = None,
        epochs_per_level: int = 20,
        top_k_fraction: float = 0.1,
    ) -> None:
        if schedule is None:
            schedule = [16, 32, 64]
        if len(schedule) < 1:
            raise ValueError("schedule must have at least one level")
        if epochs_per_level < 1:
            raise ValueError("epochs_per_level must be >= 1")
        if not (0.0 < top_k_fraction <= 1.0):
            raise ValueError(f"top_k_fraction must be in (0, 1], got {top_k_fraction}")

        self.schedule = schedule
        self.epochs_per_level = epochs_per_level
        self.top_k_fraction = top_k_fraction

    def train(self, n_vars: int, data: jnp.ndarray) -> SparseKAEMEnergy:
        """Run the full multilevel training pipeline.

        Iterates over self.schedule.  At the first level, initialises a new
        SparseKAEMEnergy.  At subsequent levels, calls _refine_to_level() to
        interpolate the univariate splines to the finer knot count, then trains
        and re-sparsifies.

        Parameters
        ----------
        n_vars : int
            Number of variables.  Must match data.shape[1].
        data : jnp.ndarray
            Training data of shape (n_data, n_vars), values in [-1, 1].

        Returns
        -------
        SparseKAEMEnergy
            Trained model at the finest resolution (schedule[-1] knots).

        Spec: REQ-SAMPLE-025
        """
        model = SparseKAEMEnergy(
            n_vars=n_vars,
            n_knots=self.schedule[0],
            top_k_fraction=self.top_k_fraction,
        )

        for i, K in enumerate(self.schedule):
            if i > 0:
                model = self._refine_to_level(model, K)
            model = self._train_level(model, data, self.epochs_per_level)
            model = self._sparsify_level(model)

        return model

    def _refine_to_level(
        self,
        model: SparseKAEMEnergy,
        K: int,
    ) -> SparseKAEMEnergy:
        """Create a new SparseKAEMEnergy at K knots with interpolated univariate splines.

        Uses KnotRefinementInterpolator to map the current model's per-variable
        control points (shape n_vars × n_knots_current) to a finer grid
        (shape n_vars × K).  The coupling matrix is re-initialised to small
        random values (not interpolated) because the coupling structure learned
        at coarse resolution may not transfer usefully to the finer spline space.

        Parameters
        ----------
        model : SparseKAEMEnergy
            Current trained model at a coarser knot count.
        K : int
            Target knot count.  Must be > model.n_knots.

        Returns
        -------
        SparseKAEMEnergy
            New model at K knots with warm-started univariate splines.

        Spec: REQ-SAMPLE-025-1
        """
        # Interpolate the univariate spline layer to finer resolution
        interpolator = KnotRefinementInterpolator(model.layer, K)
        fine_layer = interpolator.interpolate()

        # Build a fresh SparseKAEMEnergy at the new resolution
        new_model = SparseKAEMEnergy(
            n_vars=model.n_vars,
            n_knots=K,
            top_k_fraction=self.top_k_fraction,
        )
        # Inject the warm-started univariate layer
        new_model.layer = fine_layer
        return new_model

    def _train_level(
        self,
        model: SparseKAEMEnergy,
        data: jnp.ndarray,
        n_epochs: int,
    ) -> SparseKAEMEnergy:
        """Run n_epochs of score-matching + coupling gradient descent.

        Delegates to SparseKAEMEnergy.fit().  The loss history is discarded.

        Parameters
        ----------
        model : SparseKAEMEnergy
            Model to train.
        data : jnp.ndarray
            Training data, shape (n_data, n_vars).
        n_epochs : int
            Number of training epochs.

        Returns
        -------
        SparseKAEMEnergy
            The same model after in-place training.

        Spec: REQ-SAMPLE-025-2
        """
        model.fit(data, n_epochs=n_epochs)
        return model

    def _sparsify_level(self, model: SparseKAEMEnergy) -> SparseKAEMEnergy:
        """Apply sparsification to the coupling matrix after a training level.

        Calls model.sparsify() to zero out all but top-K couplings per row,
        then stores the result back into model.coupling_matrix.  This ensures
        that each new resolution level starts with a clean sparse coupling
        graph rather than inheriting noise accumulated during training.

        Parameters
        ----------
        model : SparseKAEMEnergy
            Model to sparsify in-place.

        Returns
        -------
        SparseKAEMEnergy
            The same model with coupling_matrix sparsified.

        Spec: REQ-SAMPLE-025-3
        """
        model.coupling_matrix = model.sparsify(model.coupling_matrix)
        return model
