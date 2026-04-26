"""KAN Adaptive Structure — spline grid refinement based on activation density.

**Researcher summary:**
    KAN splines have a fixed grid (num_knots control points over [-1, 1]).  If
    most activations cluster in a narrow region, the grid wastes resolution
    elsewhere.  KANAdaptiveStructure analyses per-spline activation histograms
    and doubles/halves each spline's grid where density is high/low.

**Detailed explanation for engineers:**
    After training, each spline has accumulated a distribution of input values it
    was evaluated at.  Regions visited frequently (top-2 bins > 30% of total) are
    under-resolved: more knots there would let the spline represent finer features.
    Regions almost never visited (bottom-2 bins > 60% of total) waste knots that
    could be redistributed or dropped to reduce parameters.

    restructure() builds a new KANEnergyFunction with modified num_knots per
    spline.  Because BSpline control points live in a NamedTuple (BSplineParams),
    resizing is done via linear interpolation of the old control points onto the
    new knot count — preserving the approximate learned function shape.

    This is a seed experiment: we measure whether energy_loss on a held-out eval
    set improves after restructuring + a short fine-tune.

Spec: REQ-FR11-008, SCENARIO-FR11-008
"""

from __future__ import annotations

import copy
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.models.kan import BSpline, BSplineParams, KANModel

# Density thresholds for restructuring decisions.
_HIGH_DENSITY_THRESHOLD = 0.30  # top-2 bins fraction that triggers grid doubling
_LOW_DENSITY_THRESHOLD = 0.60  # bottom-2 bins fraction that triggers grid halving
_MIN_KNOTS = 3  # never halve below this (BSpline requires >= 2)
_MAX_KNOTS = 64  # practical upper bound


def _resize_control_points(old_cp: np.ndarray, new_n_params: int) -> np.ndarray:
    """Linearly interpolate control points to a new length.

    Old control points are at positions 0, 1, ..., len-1 (uniform).  New control
    points are sampled at uniformly-spaced positions over the same range.  This
    preserves the approximate shape of the learned spline function when the grid
    is refined or coarsened.

    Args:
        old_cp: Old control point array, shape (old_n_params,).
        new_n_params: Target number of control points.

    Returns:
        Interpolated control points, shape (new_n_params,).
    """
    old_n = len(old_cp)
    old_x = np.linspace(0.0, 1.0, old_n)
    new_x = np.linspace(0.0, 1.0, new_n_params)
    return np.interp(new_x, old_x, old_cp).astype(np.float32)


class KANAdaptiveStructure:
    """Analyse KAN activation patterns and restructure spline grids accordingly.

    Workflow:
        1. Call analyze() to run the corpus through the KAN with tracking enabled.
        2. Call restructure() to build a modified KAN with refined/coarsened grids.
        3. Fine-tune the restructured KAN on the training corpus.
        4. Call evaluate_benefit() to measure energy_loss delta.

    All public methods are pure functions with respect to the KAN objects they
    receive — they do not mutate the input KAN, they return new objects.

    Spec: REQ-FR11-008
    """

    @staticmethod
    def analyze(kan: KANModel, corpus_pairs: list[Any]) -> dict[str, dict]:
        """Run corpus through KAN to build activation histograms, then classify.

        Enables activation tracking on the underlying KANEnergyFunction, runs
        every input through kan.energy(), then classifies each spline as
        high_density / low_density / neutral.

        Args:
            kan: Trained KANModel.
            corpus_pairs: Iterable of items passed to kan.energy().  Each item
                should be a JAX array of shape (input_dim,).  (In the FoVer
                seed experiment these are 1-D boolean feature vectors.)

        Returns:
            Dict of spline_id -> {"density": "high"|"low"|"neutral",
                                   "knot_count": int}
        """
        ef = kan.energy_fn
        ef.enable_activation_tracking = True
        ef._activation_histograms = {}

        for item in corpus_pairs:
            try:
                ef.energy(jnp.asarray(item, dtype=jnp.float32))
            except Exception:
                pass  # silently skip malformed inputs — corpus quality varies

        histograms = ef.get_activation_density(n_bins=20)
        ef.enable_activation_tracking = False

        analysis: dict[str, dict] = {}

        # Classify edge splines
        for (i, j), spline in ef.edge_splines.items():
            spline_id = f"edge_{i}_{j}"
            hist = histograms.get(spline_id, np.zeros(20))
            density_class = _classify_density(hist)
            analysis[spline_id] = {
                "density": density_class,
                "knot_count": spline.num_knots,
            }

        # Classify bias splines
        for idx, spline in enumerate(ef.bias_splines):
            spline_id = f"bias_{idx}"
            hist = histograms.get(spline_id, np.zeros(20))
            density_class = _classify_density(hist)
            analysis[spline_id] = {
                "density": density_class,
                "knot_count": spline.num_knots,
            }

        return analysis

    @staticmethod
    def restructure(kan: KANModel, analysis: dict[str, dict]) -> KANModel:
        """Build a new KAN with spline grids resized per analysis.

        high_density splines: num_knots doubled (add resolution where needed).
        low_density splines: num_knots halved (prune unused parameters).
        neutral splines: unchanged.

        Existing control points are linearly interpolated onto the new knot count
        so the model starts fine-tuning from a meaningful initialisation rather
        than random noise.

        Args:
            kan: Source KANModel (not mutated).
            analysis: Output of KANAdaptiveStructure.analyze().

        Returns:
            New KANModel with restructured splines and interpolated parameters.
        """
        old_ef = kan.energy_fn
        new_config = copy.deepcopy(kan.config)
        new_config.edges = list(old_ef.edges)

        # Build new KAN with same structure; we'll replace spline params below.
        new_kan = KANModel(new_config, key=jrandom.PRNGKey(42))
        new_ef = new_kan.energy_fn

        # Restructure edge splines
        for (i, j), old_spline in old_ef.edge_splines.items():
            spline_id = f"edge_{i}_{j}"
            info = analysis.get(
                spline_id, {"density": "neutral", "knot_count": old_spline.num_knots}
            )
            new_num_knots = _new_knot_count(old_spline.num_knots, info["density"])
            new_n_params = new_num_knots + old_spline.degree
            old_cp = np.array(old_spline.params.control_points)
            new_cp = _resize_control_points(old_cp, new_n_params)
            new_spline = BSpline(num_knots=new_num_knots, degree=old_spline.degree, key=None)
            new_spline.params = BSplineParams(control_points=jnp.array(new_cp))
            new_ef.edge_splines[(i, j)] = new_spline

        # Restructure bias splines
        for idx, old_spline in enumerate(old_ef.bias_splines):
            spline_id = f"bias_{idx}"
            info = analysis.get(
                spline_id, {"density": "neutral", "knot_count": old_spline.num_knots}
            )
            new_num_knots = _new_knot_count(old_spline.num_knots, info["density"])
            new_n_params = new_num_knots + old_spline.degree
            old_cp = np.array(old_spline.params.control_points)
            new_cp = _resize_control_points(old_cp, new_n_params)
            new_spline = BSpline(num_knots=new_num_knots, degree=old_spline.degree, key=None)
            new_spline.params = BSplineParams(control_points=jnp.array(new_cp))
            new_ef.bias_splines[idx] = new_spline

        return new_kan

    @staticmethod
    def evaluate_benefit(
        kan_before: KANModel,
        kan_after: KANModel,
        eval_pairs: list[Any],
    ) -> dict[str, float]:
        """Compute mean energy_loss on eval_pairs for both KANs.

        energy_loss is the mean absolute energy across the eval set — a proxy for
        how strongly the model assigns energy to the eval inputs.  Lower is
        generally better for correct inputs in a well-trained EBM (lower energy
        = higher probability).

        Args:
            kan_before: KAN before restructuring.
            kan_after: KAN after restructuring + fine-tuning.
            eval_pairs: Held-out evaluation inputs.

        Returns:
            Dict with keys: energy_loss_before, energy_loss_after, delta,
            knot_count_before, knot_count_after, knot_count_change_pct.
        """

        def _mean_energy(kan: KANModel) -> float:
            energies = []
            for item in eval_pairs:
                try:
                    e = float(kan.energy(jnp.asarray(item, dtype=jnp.float32)))
                    energies.append(abs(e))
                except Exception:
                    pass
            return float(np.mean(energies)) if energies else 0.0

        loss_before = _mean_energy(kan_before)
        loss_after = _mean_energy(kan_after)
        kc_before = kan_before.n_params
        kc_after = kan_after.n_params
        change_pct = (kc_after - kc_before) / max(kc_before, 1) * 100.0

        return {
            "energy_loss_before": loss_before,
            "energy_loss_after": loss_after,
            "delta": loss_after - loss_before,
            "knot_count_before": kc_before,
            "knot_count_after": kc_after,
            "knot_count_change_pct": change_pct,
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _classify_density(hist: np.ndarray) -> str:
    """Classify a spline as high / low / neutral based on its activation histogram.

    high_density: top-2 bins contain > 30% of activations
        (activations cluster at one end → need finer resolution there)
    low_density: bottom-2 bins contain > 60% of activations
        (most activations in low-valued region → could coarsen rest)
    neutral: everything else

    Args:
        hist: Normalized histogram, shape (n_bins,).  Must sum to ~1.

    Returns:
        "high", "low", or "neutral".
    """
    if len(hist) < 4:
        return "neutral"
    top2 = float(hist[-2] + hist[-1])
    bottom2 = float(hist[0] + hist[1])
    if top2 > _HIGH_DENSITY_THRESHOLD:
        return "high"
    if bottom2 > _LOW_DENSITY_THRESHOLD:
        return "low"
    return "neutral"


def _new_knot_count(current: int, density_class: str) -> int:
    """Compute new knot count based on density classification.

    Doubles for high_density, halves for low_density, unchanged for neutral.
    Clamped to [_MIN_KNOTS, _MAX_KNOTS].

    Args:
        current: Current num_knots.
        density_class: "high", "low", or "neutral".

    Returns:
        New num_knots.
    """
    if density_class == "high":
        new = current * 2
    elif density_class == "low":
        new = max(current // 2, 1)
    else:
        new = current
    return max(_MIN_KNOTS, min(_MAX_KNOTS, new))
