"""CIKANEnergy — Constraint-Informed KAN with boundary-concentrated spline knots.

**Why constraint-informed knot placement matters (arXiv 2412.03710):**
    Standard KAEM (Exp 447) uses *uniform* knot spacing, placing the same density of
    control points everywhere in [-1, 1].  But for constraint verification problems,
    the energy landscape is NOT uniform — it is most complex near constraint boundaries,
    where the model must sharply distinguish valid states from violated ones.  Uniform
    knots waste resolution in flat regions far from boundaries and under-resolve the
    sharp gradients exactly where they matter most.

    CIKAN (Constraint-Informed KAN) fixes this by *concentrating* extra knots within
    a window around each known constraint boundary.  The key insight: if you know a
    hard constraint fires at x=0, then the energy function must transition sharply
    from low-energy (x>0, valid) to high-energy (x<0, violated) across a narrow band
    near x=0.  Placing k-1 additional knots in that band gives the spline the degrees
    of freedom needed to represent that transition accurately.

    The cost of this approach is *reduced smoothness* far from boundaries: some regions
    of [-1, 1] have sparser coverage than the baseline.  This is the correct tradeoff
    for constraint verification, where near-boundary discrimination is the primary goal.

**How extra knots are distributed:**
    For each ConstraintBoundary at position p with sharpness s, we compute a window
    [p - s*std, p + s*std] and insert boundary_k - 1 evenly-spaced additional knots
    inside that window (on top of the base uniform grid).  The final knot array is the
    union of the base grid and all inserted knots, sorted and deduplicated.

**Relationship to KAEMEnergy:**
    CIKANLayer inherits UnivariateKAEMLayer's spline evaluation and exact sampling.
    The only change is the knot positions used at init time and when fit_with_constraints
    is called.  This keeps the exact sampling guarantee intact.

Spec: REQ-SAMPLE-025, REQ-SAMPLE-026,
      SCENARIO-SAMPLE-038, SCENARIO-SAMPLE-039, SCENARIO-SAMPLE-040
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.models.kaem_energy import KAEMEnergy, UnivariateKAEMLayer


# ---------------------------------------------------------------------------
# ConstraintBoundary
# ---------------------------------------------------------------------------


@dataclass
class ConstraintBoundary:
    """A single hard constraint boundary that CIKANLayer should concentrate knots near.

    Parameters
    ----------
    position : float
        The boundary location in the spline domain [-1, 1].  For example, 0.0 means
        the constraint fires at x=0 (split between valid/invalid half-spaces).
    sharpness : float
        Multiplier applied to the data standard deviation to define the extra-knot
        window width.  A sharpness of 1.0 means the window extends ±1 std from the
        boundary position.  Higher values add knots over a wider region.

    Why sharpness is relative to data_std rather than a fixed width:
        Different problems have different natural scales.  Anchoring the window to
        data_std makes the extra-knot region adapt to the actual data spread,
        preventing over- or under-concentration across different normalizations.
    """

    position: float
    sharpness: float = 1.0


# ---------------------------------------------------------------------------
# CIKANLayer
# ---------------------------------------------------------------------------


class CIKANLayer(UnivariateKAEMLayer):
    """Per-variable spline layer with constraint-boundary-concentrated knot density.

    Extends UnivariateKAEMLayer by replacing the uniform knot grid with one that
    has extra knots near each ConstraintBoundary.  All other behaviour (spline
    evaluation, marginal CDF, exact sampling) is inherited unchanged.

    Parameters
    ----------
    n_vars : int
        Number of variables.
    n_knots_base : int
        Number of knots in the *base* uniform grid before extra knots are added.
        Default 8.
    boundary_k : int
        Number of *total* knots in each boundary window (k-1 extra knots inserted).
        Default 4, which adds 3 extra knots near each boundary.
    key : jax.Array | None
        PRNG key for control point initialisation.

    Spec: REQ-SAMPLE-025, SCENARIO-SAMPLE-038
    """

    def __init__(
        self,
        n_vars: int,
        n_knots_base: int = 8,
        boundary_k: int = 4,
        key: jax.Array | None = None,
    ) -> None:
        # Initialize with base knot count; control_points shape matches n_knots_base.
        # When apply_boundaries() is called, n_knots and _knots are updated in place.
        super().__init__(n_vars=n_vars, n_knots=n_knots_base, key=key)
        self.n_knots_base = n_knots_base
        self.boundary_k = boundary_k

    # ------------------------------------------------------------------
    # _distribute_knots_with_boundaries
    # ------------------------------------------------------------------

    def _distribute_knots_with_boundaries(
        self,
        boundaries: Sequence[ConstraintBoundary],
        data_std: float = 0.3,
    ) -> np.ndarray:
        """Return sorted knot positions with extra density near each boundary.

        Starts from the uniform base grid of n_knots_base points in [-1, 1],
        then for each boundary inserts boundary_k - 1 additional evenly-spaced
        knots within a window of [position - sharpness*data_std, position + sharpness*data_std].

        The result is sorted, clipped to [-1, 1], and returned as a numpy array.

        Parameters
        ----------
        boundaries : sequence of ConstraintBoundary
            List of constraint boundaries to concentrate knots around.
        data_std : float
            Standard deviation of the data used to scale the window width.

        Returns
        -------
        np.ndarray
            1D sorted array of knot positions, all in [-1, 1].
            Length >= n_knots_base.

        Spec: REQ-SAMPLE-025, SCENARIO-SAMPLE-038
        """
        base_knots = list(np.linspace(-1.0, 1.0, self.n_knots_base))

        extra_knots: list[float] = []
        n_extra = self.boundary_k - 1  # boundary_k total knots in window → k-1 new interior ones
        if n_extra > 0:
            for b in boundaries:
                half_width = float(b.sharpness) * float(data_std)
                lo = float(np.clip(b.position - half_width, -1.0, 1.0))
                hi = float(np.clip(b.position + half_width, -1.0, 1.0))
                if hi > lo:
                    # linspace from lo to hi gives n_extra+2 pts; drop endpoints to avoid
                    # duplicating the boundary edges (they may already be in base_knots)
                    pts = np.linspace(lo, hi, n_extra + 2)[1:-1]
                    extra_knots.extend(float(p) for p in pts)

        all_knots = np.array(sorted(set(base_knots + extra_knots)), dtype=np.float32)
        return np.clip(all_knots, -1.0, 1.0)

    # ------------------------------------------------------------------
    # apply_boundaries
    # ------------------------------------------------------------------

    def apply_boundaries(
        self,
        boundaries: Sequence[ConstraintBoundary],
        data_std: float = 0.3,
    ) -> None:
        """Reinitialise knot positions and control points with boundary-aware spacing.

        Called by CIKANEnergy.fit_with_constraints before fitting.  Replaces the
        current knot grid with the boundary-concentrated one and reinitialises
        control_points to match the new knot count.

        Parameters
        ----------
        boundaries : sequence of ConstraintBoundary
        data_std : float

        Spec: REQ-SAMPLE-025
        """
        new_knots = self._distribute_knots_with_boundaries(boundaries, data_std)
        n_new = len(new_knots)
        self.n_knots = n_new
        self._knots = jnp.array(new_knots)
        # Reinitialise control points to match the new knot count (small random, nearly flat)
        self.control_points = jrandom.normal(jrandom.PRNGKey(1), (self.n_vars, n_new)) * 0.1


# ---------------------------------------------------------------------------
# CIKANEnergy
# ---------------------------------------------------------------------------


class CIKANEnergy(KAEMEnergy):
    """KAEM energy model with constraint-boundary-concentrated spline knots.

    Extends KAEMEnergy by replacing the uniform-knot UnivariateKAEMLayer with a
    CIKANLayer that places extra knots near user-specified constraint boundaries.
    This gives the energy function more resolution exactly where it needs it most —
    near the boundaries where valid and invalid states are separated.

    The exact sampling and differentiability properties of KAEMEnergy are preserved
    because CIKANLayer only changes knot *placement*, not the spline *form*.

    Parameters
    ----------
    n_vars : int
        Number of variables.
    n_hidden : int
        Base knot count per variable (before boundary-extra knots). Default 16.
    boundaries : list[ConstraintBoundary] | None
        Known constraint boundaries. If None, CIKANEnergy behaves identically to
        KAEMEnergy (falls back to uniform knots, no boundary concentration).
    key : jax.Array | None
        PRNG key for initialisation.

    Spec: REQ-SAMPLE-025, REQ-SAMPLE-026,
          SCENARIO-SAMPLE-038, SCENARIO-SAMPLE-039, SCENARIO-SAMPLE-040
    """

    def __init__(
        self,
        n_vars: int,
        n_hidden: int = 16,
        boundaries: list[ConstraintBoundary] | None = None,
        key: jax.Array | None = None,
    ) -> None:
        # KAEMEnergy.__init__ validates inputs and creates self.layer = UnivariateKAEMLayer.
        # We immediately replace that with a CIKANLayer so the spline evaluation path
        # runs through CIKANLayer for any subsequent fit_with_constraints call.
        super().__init__(n_vars=n_vars, n_hidden=n_hidden, key=key)

        if key is None:
            key = jrandom.PRNGKey(0)

        self.layer = CIKANLayer(n_vars=n_vars, n_knots_base=n_hidden, key=key)
        self.boundaries: list[ConstraintBoundary] = list(boundaries) if boundaries else []

    # ------------------------------------------------------------------
    # fit_with_constraints
    # ------------------------------------------------------------------

    def fit_with_constraints(
        self,
        data: jax.Array,
        boundaries: list[ConstraintBoundary],
        n_epochs: int = 100,
    ) -> list[float]:
        """Fit CIKANEnergy with boundary-aware knot placement before training.

        Sets self.boundaries, rebuilds the CIKANLayer's knot grid to concentrate
        resolution near each boundary, then calls the inherited fit() training loop.

        Parameters
        ----------
        data : jax.Array
            Training data, shape (n_data, n_vars). Values in [-1, 1].
        boundaries : list[ConstraintBoundary]
            Constraint boundaries to concentrate knots around.
        n_epochs : int
            Training epochs. Default 100.

        Returns
        -------
        list[float]
            Loss history from fit().

        Spec: REQ-SAMPLE-025, REQ-SAMPLE-026, SCENARIO-SAMPLE-040
        """
        self.boundaries = list(boundaries)

        data_np = np.array(data)
        data_std = float(np.std(data_np))
        if data_std < 1e-6:
            data_std = 0.3

        # Reinitialise knots with boundary concentration before training
        self.layer.apply_boundaries(boundaries, data_std=data_std)

        # Sync n_hidden to the new (potentially larger) knot count so fit() loop works
        self.n_hidden = self.layer.n_knots

        return self.fit(data, n_epochs=n_epochs)
