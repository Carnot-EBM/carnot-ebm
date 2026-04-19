"""Hallucination Basin Detector — basin depth from hidden-state trajectories.

**Researcher summary:**
    Implements the basin-depth hallucination signal from arXiv 2604.04743.
    Correct reasoning follows deep-basin attractors in LLM latent space;
    hallucinated reasoning drifts into shallow basins with high escape probability.
    Basin depth is estimated via finite-difference energy perturbation — no
    additional model passes required beyond the energy proxy callable.

**Detailed explanation for engineers:**
    The paper arXiv 2604.04743 shows that LLM hidden states during correct
    reasoning sit in deep energy basins (stable attractors), while hallucinated
    hidden states sit in shallow basins or near saddle points (high escape
    probability).

    For each hidden state x_t in a sequence:

        energy_at_x = energy_fn(x_t)
        perturbed_energies = [energy_fn(x_t + scale * noise_i) for i in range(k)]
        depth(t) = energy_at_x - min(perturbed_energies)

    A positive depth means we are at a local minimum — perturbations INCREASE
    energy, confirming a stable basin.  Near-zero or negative depth means the
    current point is a saddle or on a slope — some perturbation decreases energy,
    indicating an unstable region (high hallucination risk).

    Basin risk score:
        mean_depth = mean(depth_t over T timesteps)
        basin_risk_score = 1.0 - sigmoid(mean_depth)

    So:
        deep basin  (mean_depth >> 0) → sigmoid → high → 1 - high → LOW risk score
        shallow basin (mean_depth ≈ 0) → sigmoid → 0.5 → 1 - 0.5 → HIGH risk score

    This positions the HallucinationBasinDetector as Tier 0d in the verification
    cascade: after SpilledEnergy (Tier 0b, logit-space signal) but before the
    heavier SinkProbe (Tier 1).

    **CI usage:** Pass any callable as `energy_fn`.  For unit tests, a simple
    quadratic `lambda x: float(jnp.sum(x**2))` works without a trained EBM.

Spec: REQ-VERIFY-107, REQ-VERIFY-108,
      SCENARIO-VERIFY-140, SCENARIO-VERIFY-141, SCENARIO-VERIFY-142
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BasinEstimate:
    """Result of HallucinationBasinDetector.detect().

    **Researcher summary:**
        Aggregates basin depth, escape probability, and risk score for a
        hidden-state trajectory.  basin_risk_score is the primary signal:
        > 0.5 means shallow basin (hallucination risk), < 0.5 means deep
        basin (correct reasoning likely).

    **Detailed explanation for engineers:**
        - basin_depth: mean depth over the T hidden states.  Positive = stable
          minimum; near-zero = saddle or flat region.
        - escape_probability: sigmoid(-basin_depth).  High = easy to escape
          the current energy state = unstable reasoning.
        - basin_risk_score: 1.0 - sigmoid(basin_depth).  In [0, 1].  High
          score = shallow basin = hallucination-prone.

    Spec: REQ-VERIFY-108
    """

    basin_depth: float
    escape_probability: float
    basin_risk_score: float


# ---------------------------------------------------------------------------
# estimate_basin_depth — single hidden-state depth estimation
# ---------------------------------------------------------------------------


def estimate_basin_depth(
    hidden_state: jax.Array,
    energy_fn: Callable[[jax.Array], float],
    n_perturbations: int = 8,
    perturbation_scale: float = 0.1,
    *,
    rng_seed: int = 0,
) -> float:
    """Estimate basin depth for a single hidden state via finite-difference perturbation.

    **Detailed explanation for engineers:**
        For a hidden state x sitting at a local energy minimum, any perturbation
        moves to higher energy → depth is positive and large.  For a hidden state
        on a saddle or slope, some perturbations move to LOWER energy → depth is
        near zero or negative.

        Algorithm:
            energy_at_x = energy_fn(x)
            for i in range(n_perturbations):
                noise = perturbation_scale * N(0, I)  # isotropic Gaussian
                perturbed_energy = energy_fn(x + noise)
            depth = min(perturbed_energies) - energy_at_x  # positive = deeper basin

        The key insight: depth > 0 iff the current point is a local minimum
        relative to its Gaussian neighbourhood.  The larger the depth, the
        harder it is to escape the basin.

    Args:
        hidden_state: 1-D JAX array representing one LLM hidden state.
        energy_fn: Callable mapping a hidden state to a scalar energy value.
            Can be any function — a trained EBM, a simple quadratic proxy, etc.
        n_perturbations: Number of random perturbation directions to sample.
            More samples = more stable depth estimate at the cost of energy_fn
            evaluations.  Default 8 is fast and sufficient for CI tests.
        perturbation_scale: Standard deviation of the isotropic Gaussian noise.
            Should be much smaller than the length scale of the energy landscape.
            Default 0.1 works for normalised hidden states.
        rng_seed: JAX PRNG seed.  Deterministic by default for reproducibility.

    Returns:
        Basin depth as a float.  Positive = local minimum, near-zero = saddle.

    Spec: REQ-VERIFY-107, SCENARIO-VERIFY-140, SCENARIO-VERIFY-141
    """
    energy_at_x = energy_fn(hidden_state)

    key = jax.random.PRNGKey(rng_seed)
    perturbed_min = float("inf")

    for i in range(n_perturbations):
        key, subkey = jax.random.split(key)
        noise = perturbation_scale * jax.random.normal(subkey, shape=hidden_state.shape)
        perturbed_energy = energy_fn(hidden_state + noise)
        if perturbed_energy < perturbed_min:
            perturbed_min = perturbed_energy

    # depth = min(perturbed) - energy_at_x:
    # positive means all perturbations raised energy = deep basin (hard to escape).
    # negative means some perturbation lowered energy = shallow basin (easy to escape).
    return perturbed_min - float(energy_at_x)


# ---------------------------------------------------------------------------
# HallucinationBasinDetector
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid for scalar inputs."""
    if x >= 0.0:
        return 1.0 / (1.0 + float(jnp.exp(-x)))
    exp_x = float(jnp.exp(x))
    return exp_x / (1.0 + exp_x)


class HallucinationBasinDetector:
    """Tier 0d hallucination signal: basin depth from hidden-state trajectories.

    **Researcher summary:**
        Estimates whether a sequence of LLM hidden states sits in a deep energy
        basin (correct reasoning) or a shallow basin/saddle (hallucination risk).
        Positioned after SpilledEnergy (Tier 0b, logit-space) and before SinkProbe
        (Tier 1, attention-sink).  Requires a callable energy function but no
        additional model forward passes.

    **Detailed explanation for engineers:**
        Pipeline position:
            Tier 0a (fast pre-screen) →
            Tier 0b (SpilledEnergy) →
            Tier 0d (HallucinationBasinDetector) ← THIS CLASS
            Tier 1 (SinkProbe) → Tier 2 (EORM) → Tier 3 (Ising)

        detect() processes a 2-D array of hidden states (T, D):
            - Estimates basin depth at each of T timesteps.
            - Computes mean depth over the trajectory.
            - Maps mean depth to a basin_risk_score in [0, 1].

        benchmark() evaluates AUROC of basin_risk_score against a binary label
        (1 = hallucinated, 0 = correct) on a list of (hidden_states, label) pairs.

        **Energy function (energy_fn):** Any callable `(jax.Array) -> float`.
        For production use, pass an IsingEBM.energy() or equivalent trained model.
        For CI tests, pass a simple quadratic proxy.

    Attributes:
        energy_fn: Energy proxy callable.  Maps a D-dimensional hidden state to
            a scalar float.  Called O(T * n_perturbations) times per detect() call.
        n_perturbations: Number of perturbation samples per timestep.  Default 8.
        threshold: basin_risk_score threshold for flagging.  Not used by detect()
            itself but available for downstream thresholding.  Default 0.0.

    Spec: REQ-VERIFY-108, SCENARIO-VERIFY-142
    """

    def __init__(
        self,
        energy_fn: Callable[[jax.Array], float],
        n_perturbations: int = 8,
        threshold: float = 0.0,
        perturbation_scale: float = 0.1,
    ) -> None:
        """Create a HallucinationBasinDetector.

        Args:
            energy_fn: Energy proxy.  Any callable mapping a 1-D JAX array to a
                float.  Evaluated 1 + n_perturbations times per hidden state.
            n_perturbations: Random perturbations per hidden state.  Default 8.
            threshold: Optional flagging threshold on basin_risk_score.
                A score above threshold indicates hallucination risk.  Default 0.0
                (unused by detect; provided for integration convenience).
            perturbation_scale: Gaussian noise scale for perturbations.  Default 0.1.
        """
        self.energy_fn = energy_fn
        self.n_perturbations = n_perturbations
        self.threshold = threshold
        self.perturbation_scale = perturbation_scale

    def detect(
        self,
        hidden_states: jax.Array,
        *,
        rng_seed: int = 0,
    ) -> BasinEstimate:
        """Estimate basin depth and hallucination risk for a hidden-state trajectory.

        **Detailed explanation for engineers:**
            hidden_states has shape (T, D) where T = sequence length and D = hidden dim.
            Each row is one timestep's hidden state.  We estimate basin depth at each
            timestep and aggregate to a single risk score.

            Why mean depth rather than min depth?
            Mean is more robust to occasional saddle-point timesteps in otherwise
            deep-basin trajectories.  A single shallow timestep should not flip the
            verdict for an otherwise stable sequence.

        Args:
            hidden_states: JAX array of shape (T, D) — T hidden states of dimension D.
                If 1-D (shape D), treated as T=1.
            rng_seed: Base PRNG seed.  Each timestep uses seed + timestep_index to
                ensure independent noise samples across timesteps.

        Returns:
            BasinEstimate with basin_depth, escape_probability, basin_risk_score.

        Spec: REQ-VERIFY-108, SCENARIO-VERIFY-142
        """
        arr = jnp.asarray(hidden_states)
        if arr.ndim == 1:
            arr = arr[None, :]  # treat as (1, D)

        T = arr.shape[0]
        depths = []
        for t in range(T):
            d = estimate_basin_depth(
                arr[t],
                self.energy_fn,
                n_perturbations=self.n_perturbations,
                perturbation_scale=self.perturbation_scale,
                rng_seed=rng_seed + t,
            )
            depths.append(d)

        mean_depth = sum(depths) / T
        escape_probability = _sigmoid(-mean_depth)
        basin_risk_score = 1.0 - _sigmoid(mean_depth)

        return BasinEstimate(
            basin_depth=mean_depth,
            escape_probability=escape_probability,
            basin_risk_score=basin_risk_score,
        )

    def benchmark(
        self,
        responses: list[tuple[jax.Array, int]],
        labels: list[int] | None = None,
    ) -> dict[str, float]:
        """Evaluate AUROC of basin_risk_score against binary hallucination labels.

        **Detailed explanation for engineers:**
            responses can be:
              - list of (hidden_states, label) tuples (labels is then ignored)
              - list of hidden_states arrays (labels must be provided separately)

            Returns a dict with key 'auroc' in [0, 1].  AUROC > 0.5 means the
            basin_risk_score is better than random at predicting hallucination.

        Args:
            responses: List of (hidden_states_array, label) tuples, or list of
                hidden_states arrays if labels is provided separately.
            labels: Optional separate label list.  Used only when responses is a
                list of arrays (not tuples).

        Returns:
            Dict with 'auroc' key.  Falls back to 0.5 if sklearn is unavailable
            or if only one class is present in the labels.

        Spec: REQ-VERIFY-108
        """
        # Unpack (hidden_states, label) tuples OR (arrays + separate labels).
        if labels is None:
            pairs = [(jnp.asarray(hs), int(lbl)) for hs, lbl in responses]
        else:
            pairs = [(jnp.asarray(hs), int(lbl)) for hs, lbl in zip(responses, labels)]

        y_true = [lbl for _, lbl in pairs]
        y_score = [self.detect(hs).basin_risk_score for hs, _ in pairs]

        try:
            from sklearn.metrics import roc_auc_score  # noqa: PLC0415

            if len(set(y_true)) < 2:
                return {"auroc": 0.5}
            return {"auroc": float(roc_auc_score(y_true, y_score))}
        except Exception:
            return {"auroc": 0.5}
