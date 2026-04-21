"""MultilevelKAEMTrainer — Multilevel KAN training via knot refinement schedule.

**Why multilevel training works (arXiv 2603.04827, March 2026):**
    Training a KAN spline directly at high knot resolution (e.g. K=256) starts
    in a severely over-parameterised regime where gradient descent can stagnate
    in poor local minima.  Multilevel training instead:

    1. Trains at coarse resolution (K=16) where the landscape is smooth and
       global structure is easy to find.
    2. Interpolates the coarse control points analytically to a finer grid (K=32),
       warm-starting fine training near the global minimum.
    3. Repeats the interpolation and training until the final resolution (K=128).

    This is the same principle as multigrid methods in numerical PDE solving:
    coarse grids capture long-wavelength structure cheaply, fine grids refine.

**What KnotRefinementInterpolator does:**
    Given a coarse UnivariateKAEMLayer with K_c knot positions and control points,
    it creates a new layer with K_f > K_c knot positions by linearly interpolating
    the coarse control points.  Linear interpolation is used instead of cubic spline
    interpolation to match the KAEM spline evaluation kernel (which is also linear).

**Why this addresses RETRO-057 (LowRankKAEM accuracy gap):**
    RETRO-057 noted that LowRankKAEMEnergy has energy_mad_normalized >> 0.05
    even after affine calibration at small k.  One root cause is that the full-rank
    KAEMEnergy itself may be stuck in a poor local minimum due to over-parameterised
    early training.  Multilevel training improves the base KAEM accuracy BEFORE any
    rank reduction, giving the calibration layer a better reference to correct from.

Spec: REQ-SAMPLE-038, SCENARIO-SAMPLE-063, SCENARIO-SAMPLE-064
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from carnot.models.kaem_energy import KAEMEnergy, UnivariateKAEMLayer


# ---------------------------------------------------------------------------
# KnotRefinementInterpolator
# ---------------------------------------------------------------------------


class KnotRefinementInterpolator:
    """Create a finer UnivariateKAEMLayer by linearly interpolating coarse control points.

    This is the geometric interpolation operator between KAN resolution levels,
    as described in arXiv 2603.04827 Section 3.  Linear interpolation is used
    because KAEM's spline evaluation is also piecewise-linear (not cubic), so
    the interpolated control points are consistent with what a finer-resolution
    layer would learn if initialised from scratch near the coarse solution.

    Parameters
    ----------
    coarse_layer : UnivariateKAEMLayer
        The trained coarse-resolution layer to upsample.
    fine_n_knots : int
        Target knot count for the fine-resolution layer.  Must be > coarse_layer.n_knots.
    """

    def __init__(
        self,
        coarse_layer: UnivariateKAEMLayer,
        fine_n_knots: int,
    ) -> None:
        if fine_n_knots <= coarse_layer.n_knots:
            raise ValueError(
                f"fine_n_knots ({fine_n_knots}) must be > coarse n_knots ({coarse_layer.n_knots})"
            )
        self.coarse = coarse_layer
        self.fine_n = fine_n_knots

    def interpolate(self) -> UnivariateKAEMLayer:
        """Create a fine-resolution layer by linear interpolation of coarse control points.

        The coarse knot positions are uniformly spaced in [-1, 1].  We create
        fine_n uniformly spaced knot positions in the same range, then for each
        variable's control-point array, use numpy linear interpolation to map
        coarse values to fine positions.

        The interpolation preserves the energy landscape shape: a coarse spline
        with value v at position x_c will produce a fine spline with value ~v
        at the nearest fine knot.  This gives a warm start that is much closer
        to the global minimum than random initialisation at fine resolution.

        Returns
        -------
        UnivariateKAEMLayer
            New layer with fine_n knots, control points set by interpolation.
        """
        coarse_knots = np.linspace(-1.0, 1.0, self.coarse.n_knots)
        fine_knots = np.linspace(-1.0, 1.0, self.fine_n)

        coarse_ctrl = np.array(self.coarse.control_points)  # (n_vars, K_c)
        n_vars = self.coarse.n_vars

        fine_ctrl = np.zeros((n_vars, self.fine_n), dtype=np.float32)
        for i in range(n_vars):
            # numpy interp: fine_knots are xp query points; coarse_ctrl[i] are fp values
            fine_ctrl[i] = np.interp(fine_knots, coarse_knots, coarse_ctrl[i])

        # Build the fine layer and inject interpolated control points
        fine_layer = UnivariateKAEMLayer(n_vars=n_vars, n_knots=self.fine_n)
        fine_layer.control_points = jnp.array(fine_ctrl)
        return fine_layer


# ---------------------------------------------------------------------------
# MultilevelKAEMTrainer
# ---------------------------------------------------------------------------


class MultilevelKAEMTrainer:
    """Train KAEMEnergy using a multilevel knot refinement schedule.

    Instead of training a single high-resolution model from scratch (which can
    get stuck in poor local minima), this trainer:

    1. Initialises at coarse resolution schedule[0] (e.g. K=16).
    2. Trains for epochs_per_level epochs using KAEMEnergy.fit().
    3. Interpolates the learned control points to schedule[1] (e.g. K=32)
       using KnotRefinementInterpolator.
    4. Continues training and interpolating up the resolution ladder.

    The total number of training epochs is len(schedule) * epochs_per_level.
    For the default schedule [16, 32, 64, 128] with epochs_per_level=20,
    this equals 80 epochs — the same budget as a naive K=128 baseline.

    **Why equal epoch budget is the right comparison:**
        To be a fair comparison, multilevel training should not get MORE gradient
        steps than the baseline.  The schedule is designed so that the total
        number of per-variable gradient updates is equal, but the updates at
        each level are more informative because the model starts near the basin
        of the global minimum from the coarser level.

    Parameters
    ----------
    schedule : list[int]
        Knot counts per level in increasing order.  Default [16, 32, 64, 128].
    epochs_per_level : int
        Number of training epochs at each resolution level.  Default 20.
    """

    def __init__(
        self,
        schedule: list[int] | None = None,
        epochs_per_level: int = 20,
    ) -> None:
        if schedule is None:
            schedule = [16, 32, 64, 128]
        if len(schedule) < 1:
            raise ValueError("schedule must have at least one level")
        if epochs_per_level < 1:
            raise ValueError("epochs_per_level must be >= 1")

        self.schedule = schedule
        self.epochs_per_level = epochs_per_level

    def train(self, n_vars: int, data: jnp.ndarray) -> KAEMEnergy:
        """Run the full multilevel training pipeline and return the final KAEMEnergy.

        At each resolution level:
        - If first level: create a new KAEMEnergy at schedule[0] knots.
        - Otherwise: use KnotRefinementInterpolator to warm-start from the
          previous level's trained layer.
        - Call _train_level() to run epochs_per_level gradient steps.

        Parameters
        ----------
        n_vars : int
            Number of variables (must match data.shape[1]).
        data : jnp.ndarray
            Training data of shape (n_data, n_vars), values in [-1, 1].

        Returns
        -------
        KAEMEnergy
            Trained model at the finest resolution (schedule[-1] knots).
        """
        model = KAEMEnergy(n_vars=n_vars, n_hidden=self.schedule[0])

        for i, K in enumerate(self.schedule):
            if i > 0:
                # Warm-start the next resolution level by interpolating weights
                interpolator = KnotRefinementInterpolator(model.layer, K)
                fine_layer = interpolator.interpolate()
                # Rebuild KAEMEnergy shell with correct n_hidden, inject fine layer
                new_model = KAEMEnergy(n_vars=n_vars, n_hidden=K)
                new_model.layer = fine_layer
                model = new_model

            model = self._train_level(model, data, self.epochs_per_level)

        return model

    def _train_level(
        self,
        model: KAEMEnergy,
        data: jnp.ndarray,
        n_epochs: int,
    ) -> KAEMEnergy:
        """Run n_epochs of score-matching gradient descent at the current resolution.

        Delegates to KAEMEnergy.fit() which implements per-variable marginal
        score matching.  The loss history is discarded; only the trained model
        state is returned.

        Parameters
        ----------
        model : KAEMEnergy
            Model to train in-place (JAX arrays are mutated via .at[].set()).
        data : jnp.ndarray
            Training data, shape (n_data, n_vars).
        n_epochs : int
            Number of epochs to run.

        Returns
        -------
        KAEMEnergy
            The same model object after training (returned for clarity).
        """
        model.fit(data, n_epochs=n_epochs)
        return model
