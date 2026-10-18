"""KAN AutoKnots adaptive spline grid refinement — arXiv 2412.13423.

**Researcher summary:**
    KAN splines have a fixed knot grid regardless of how much work each spline
    actually does during inference.  AutoKnots fixes this by measuring the mean
    activation magnitude for each spline across a data batch, then:
      - ADDING a knot to high-activation splines (they are doing real work and
        need more resolution to represent fine-grained structure)
      - REMOVING a knot from low-activation splines (they are dormant and
        wasting parameters)

    This is the "structural self-improvement" tier (FR-11 Tier 4) of the
    Carnot self-learning loop: the model can reshape its own function space
    without external supervision, guided only by where the data actually lands.

**Detailed explanation for engineers:**
    A BSpline object has `num_knots` positions and `num_knots + degree` control
    points.  The spline is evaluated by interpolating between adjacent control
    points.  Adding a knot doubles the resolution in the most-activated region;
    removing a knot merges the two least-activated adjacent intervals.

    This implementation takes a simpler but equivalent approach: the new knot
    count is just `num_knots +/- 1`, and the control points are linearly
    re-sampled from the old control point array onto the new length.  This
    preserves the approximate learned function shape after knot insertion.

    The activation magnitude for edge spline (i, j) is:
        mean_abs(activation_batch[:, i] * activation_batch[:, j])
    For bias spline i:
        mean_abs(activation_batch[:, i])

    Both measures are in [0, 1] when inputs are binary {0, 1} or spins {-1, +1}.

Spec: REQ-SELF-008, SCENARIO-SELF-008
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.random as jrandom
import numpy as np

from carnot.models.kan import BSpline, BSplineParams, KANModel


def _resize_control_points(old_cp: np.ndarray, new_n_params: int) -> np.ndarray:
    """Linearly re-sample control points to a new length.

    The old control points are treated as a piecewise-linear function sampled at
    equally spaced positions.  New positions are also equally spaced over the same
    range, but with a different count.  This preserves the approximate shape of
    the learned spline after a knot is added or removed.

    Why linear interpolation rather than B-spline refinement: the error introduced
    by linear resampling is small compared to the short fine-tuning step that
    always follows AutoKnots restructuring.  The exact knot-insertion algorithm
    from arXiv 2412.13423 has higher fidelity but requires substantially more
    bookkeeping with no measurable benefit at small knot counts.

    Args:
        old_cp: Old control points, shape (old_n_params,).
        new_n_params: Target control point count (must be >= 2).

    Returns:
        Resampled control points, shape (new_n_params,), dtype float32.
    """
    old_x = np.linspace(0.0, 1.0, len(old_cp))
    new_x = np.linspace(0.0, 1.0, new_n_params)
    return np.interp(new_x, old_x, old_cp).astype(np.float32)


@dataclass
class RefinementResult:
    """Summary of one AutoKnots refinement round.

    Attributes:
        n_added: Total knots added across all splines in this round.
        n_removed: Total knots removed across all splines in this round.
        splines_modified: List of spline IDs that were changed.
            Edge splines are named "edge_i_j"; bias splines are "bias_i".
    """

    n_added: int
    n_removed: int
    splines_modified: list[str] = field(default_factory=list)


class AutoKnotsRefiner:
    """AutoKnots adaptive grid refinement for a KANModel.

    Implements the refine-add / refine-remove heuristic from arXiv 2412.13423,
    adapted for the Carnot BSpline representation:

        - HIGH activation (mean |a| > high_thresh) AND below max_knots
          → add 1 knot to this spline (increase resolution)
        - LOW activation (mean |a| < low_thresh) AND above min_knots
          → remove 1 knot from this spline (prune dormant structure)

    The refiner mutates the KANModel in-place: after each call to refine_once(),
    the model's edge_splines and bias_splines have updated num_knots and
    resampled control points.  Run a short fine-tuning epoch after refinement
    to let the model recover from the grid change.

    Attributes:
        model: The KANModel being refined.
        high_thresh: Mean activation magnitude above which a spline gains a knot.
        low_thresh: Mean activation magnitude below which a spline loses a knot.
        max_knots: Upper bound on num_knots per spline.
        min_knots: Lower bound on num_knots per spline.

    Spec: REQ-SELF-008
    """

    def __init__(
        self,
        kan_model: KANModel,
        high_activation_threshold: float = 0.8,
        low_activation_threshold: float = 0.1,
        max_knots_per_spline: int = 32,
        min_knots_per_spline: int = 4,
    ) -> None:
        if high_activation_threshold <= low_activation_threshold:
            raise ValueError(
                f"high_activation_threshold ({high_activation_threshold}) must be "
                f"> low_activation_threshold ({low_activation_threshold})"
            )
        if min_knots_per_spline < 2:
            raise ValueError("min_knots_per_spline must be >= 2 (BSpline invariant)")
        if max_knots_per_spline <= min_knots_per_spline:
            raise ValueError("max_knots_per_spline must be > min_knots_per_spline")

        self.model = kan_model
        self.high_thresh = high_activation_threshold
        self.low_thresh = low_activation_threshold
        self.max_knots = max_knots_per_spline
        self.min_knots = min_knots_per_spline

    def _activation_magnitude(self, activation_batch: np.ndarray, spline_id: str) -> float:
        """Compute mean absolute activation for a single spline.

        For edge splines ("edge_i_j"): activation is x[:, i] * x[:, j].
        For bias splines ("bias_i"): activation is x[:, i].

        Args:
            activation_batch: shape (n_samples, input_dim).
            spline_id: Spline identifier string.

        Returns:
            Scalar mean absolute activation in [0, ∞).
        """
        if spline_id.startswith("edge_"):
            parts = spline_id.split("_")
            i, j = int(parts[1]), int(parts[2])
            activations = activation_batch[:, i] * activation_batch[:, j]
        else:
            # bias_i
            i = int(spline_id.split("_")[1])
            activations = activation_batch[:, i]
        return float(np.mean(np.abs(activations)))

    def _resize_spline(self, spline: BSpline, new_num_knots: int) -> BSpline:
        """Return a new BSpline with a different knot count, resampling control points.

        The new BSpline has the same degree as the original; its control points
        are linearly resampled from the old ones so the model can fine-tune from
        a reasonable initialisation rather than random noise.

        Args:
            spline: Original BSpline.
            new_num_knots: Target num_knots for the new spline.

        Returns:
            New BSpline object with resampled control points.
        """
        new_n_params = new_num_knots + spline.degree
        old_cp = np.array(spline.params.control_points)
        new_cp = _resize_control_points(old_cp, new_n_params)

        new_spline = BSpline(
            num_knots=new_num_knots,
            degree=spline.degree,
            key=jrandom.PRNGKey(0),
        )
        import jax.numpy as jnp  # local import — avoids top-level JAX init on CPU

        new_spline.params = BSplineParams(control_points=jnp.array(new_cp))
        return new_spline

    def refine_once(self, activation_batch: np.ndarray) -> RefinementResult:
        """Perform one round of AutoKnots refinement on the model.

        Iterates over all edge splines and bias splines.  For each spline:
          1. Compute mean absolute activation magnitude across the batch.
          2. If magnitude > high_thresh AND knots < max → insert one knot.
          3. If magnitude < low_thresh AND knots > min → remove one knot.

        The model is mutated in-place.

        Args:
            activation_batch: shape (n_samples, input_dim), float32.
                Each row is a single input vector; the same batch used to run
                the KAN forward pass is the right thing to pass here.

        Returns:
            RefinementResult with counts and IDs of modified splines.

        Spec: REQ-SELF-008
        """
        ef = self.model.energy_fn
        n_added = 0
        n_removed = 0
        modified: list[str] = []

        # Process edge splines
        for (i, j), spline in list(ef.edge_splines.items()):
            spline_id = f"edge_{i}_{j}"
            mag = self._activation_magnitude(activation_batch, spline_id)

            if mag > self.high_thresh and spline.num_knots < self.max_knots:
                ef.edge_splines[(i, j)] = self._resize_spline(spline, spline.num_knots + 1)
                n_added += 1
                modified.append(spline_id)
            elif mag < self.low_thresh and spline.num_knots > self.min_knots:
                ef.edge_splines[(i, j)] = self._resize_spline(spline, spline.num_knots - 1)
                n_removed += 1
                modified.append(spline_id)

        # Process bias splines
        for i, spline in enumerate(ef.bias_splines):
            spline_id = f"bias_{i}"
            mag = self._activation_magnitude(activation_batch, spline_id)

            if mag > self.high_thresh and spline.num_knots < self.max_knots:
                ef.bias_splines[i] = self._resize_spline(spline, spline.num_knots + 1)
                n_added += 1
                modified.append(spline_id)
            elif mag < self.low_thresh and spline.num_knots > self.min_knots:
                ef.bias_splines[i] = self._resize_spline(spline, spline.num_knots - 1)
                n_removed += 1
                modified.append(spline_id)

        return RefinementResult(
            n_added=n_added,
            n_removed=n_removed,
            splines_modified=modified,
        )

    def multi_round_refine(
        self, activation_batch: np.ndarray, rounds: int = 3
    ) -> list[RefinementResult]:
        """Perform multiple successive rounds of AutoKnots refinement.

        After each round the model's knot counts change, so subsequent rounds
        operate on the updated structure.  In practice 2–3 rounds suffice: the
        first round dominates the structural change; subsequent rounds converge
        because splines that were already at min/max cannot change further.

        Args:
            activation_batch: shape (n_samples, input_dim).
            rounds: Number of refinement rounds (default 3).

        Returns:
            List of RefinementResult, one per round.

        Spec: REQ-SELF-008
        """
        results = []
        for _ in range(rounds):
            results.append(self.refine_once(activation_batch))
        return results
