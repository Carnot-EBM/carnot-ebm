"""capo_loss — CAPO calibration-aware contrastive loss for JEPA v13.

**Why this exists (CAPO, arXiv 2604.12632):**
    JEPA v12 (Exp 607) achieved OOD AUC=0.5 despite near-perfect in-distribution
    performance.  Post-mortem: the model output extreme confidence scores (near 0 or 1)
    — a classic sign of miscalibration.  Miscalibrated models discriminate correctly on
    familiar data but collapse on unseen distributions because they learned to rely on
    spurious surface cues rather than genuine correctness signals.

    CAPO (Calibrated Accuracy Preference Optimization) co-optimises two objectives:
        1. Contrastive margin loss: pushes the energy of incorrect responses above the
           energy of correct responses by at least `margin` units.  This is the same
           CPMI objective used in v12, so v13 retains the discriminative power.
        2. Expected Calibration Error (ECE): penalises the model when its *confidence*
           (sigmoid(-energy)) is not proportional to its *accuracy*.  A perfectly
           calibrated model assigns probability 0.7 to a bucket of responses and is
           right 70% of the time on that bucket.

    By jointly minimising both, v13 is forced to score responses with probabilities
    that reflect genuine uncertainty rather than learned surface-pattern confidence.

**Why ECE and not Temperature Scaling?**
    Temperature scaling is post-hoc — it cannot fix a fundamentally biased energy
    landscape.  ECE as a training loss forces calibration during gradient descent,
    which changes what the model learns rather than just rescaling the output.

**Why binned ECE instead of distributional calibration?**
    Binned ECE (equal-width bins over [0,1]) is the standard metric for classification
    calibration (Guo et al., 2017).  It is differentiable enough for JAX's grad when
    we use hard-bin assignment — JAX propagates gradients through per-bin means,
    which is where the model can actually change its behaviour.

Spec: REQ-VERIFY-120, REQ-VERIFY-121,
      SCENARIO-VERIFY-157, SCENARIO-VERIFY-158, SCENARIO-VERIFY-159
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def ece_loss(
    predicted_probs: jnp.ndarray,
    labels: jnp.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error: mean |confidence - accuracy| over probability bins.

    **Layman explanation:**
        Split all predictions into n_bins equally spaced buckets from 0% to 100%
        confidence.  For each bucket, compute how often the model is *actually* right
        on those examples.  ECE = average gap between "what the model claimed" and
        "what was actually true."  ECE=0 means perfect calibration.  ECE=0.10 means
        a 10% average miscalibration gap.

    **Why hard bins:**
        Hard bins via boolean masks produce zero gradient through bin *assignment*
        (JAX stop-gradient at boolean ops), but the gradient still flows through the
        per-bin mean — which is what the model can change.  This is sufficient for
        ECE to act as a regularisation signal.

    Args:
        predicted_probs: jnp.ndarray of shape (N,), values in [0, 1].
                         Interpreted as P(response is incorrect).
        labels:          jnp.ndarray of shape (N,), values in {0, 1}.
                         1 = actually incorrect, 0 = actually correct.
        n_bins:          Number of equal-width bins over [0, 1].  Default 10.

    Returns:
        Scalar float: mean absolute calibration error over non-empty bins.
        Returns 0.0 when N=0 (safe for empty batches).

    Spec: REQ-VERIFY-121
    """
    if predicted_probs.shape[0] == 0:
        return jnp.array(0.0)

    # Use numpy (not jnp) so bin edges are concrete Python floats — required for
    # comparison operators inside jax.jit where traced arrays cannot be used as
    # Python scalars via float().
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    labels_float = labels.astype(jnp.float32)

    total_ece = jnp.array(0.0)
    n_nonempty = jnp.array(0)

    for i in range(n_bins):
        lo = float(bin_edges[i])
        hi = float(bin_edges[i + 1])
        # Include upper bound in last bin to capture prob=1.0 exactly.
        if i < n_bins - 1:
            in_bin = (predicted_probs >= lo) & (predicted_probs < hi)
        else:
            in_bin = (predicted_probs >= lo) & (predicted_probs <= hi)

        bin_count = jnp.sum(in_bin.astype(jnp.float32))
        is_nonempty = bin_count > 0

        mean_conf = jnp.where(
            is_nonempty,
            jnp.sum(jnp.where(in_bin, predicted_probs, 0.0)) / jnp.maximum(bin_count, 1.0),
            0.0,
        )
        mean_acc = jnp.where(
            is_nonempty,
            jnp.sum(jnp.where(in_bin, labels_float, 0.0)) / jnp.maximum(bin_count, 1.0),
            0.0,
        )
        total_ece = total_ece + jnp.where(is_nonempty, jnp.abs(mean_conf - mean_acc), 0.0)
        n_nonempty = n_nonempty + jnp.where(is_nonempty, 1, 0)

    return jnp.where(n_nonempty > 0, total_ece / n_nonempty.astype(jnp.float32), 0.0)


def capo_loss(
    energy_scores: jnp.ndarray,
    labels: jnp.ndarray,
    margin: float = 1.0,
    lambda_calib: float = 0.1,
) -> float:
    """CAPO combined loss: contrastive margin + calibration penalty.

    **Layman explanation:**
        The model outputs an energy score for each response — higher energy means
        the model thinks the response is more likely to be wrong.  Two objectives:

        1. **Contrastive loss**: for every (correct, incorrect) pair in the batch,
           the incorrect response's energy should be at least `margin` higher than
           the correct response's energy.  If not, penalise by the shortfall.
           Hinge form: max(0, margin - (E_wrong - E_right)).

        2. **Calibration loss**: convert energies to probabilities via sigmoid(energy).
           Then compute ECE — how miscalibrated those probabilities are vs. actual
           labels.  Penalise by `lambda_calib * ECE`.

        Combining these ensures the model both *ranks* responses correctly AND assigns
        *well-calibrated* probabilities.  v12 achieved ranking but not calibration,
        which caused OOD collapse (extreme outputs, no uncertainty).

    **Pairing strategy:**
        All-vs-all within the batch: every correct example is paired against every
        incorrect example.  This gives dense gradient signal regardless of whether
        the batch is from a single question or mixed questions.

    Args:
        energy_scores:  jnp.ndarray of shape (N,), unbounded floats.
                        Higher energy = model believes response is incorrect.
        labels:         jnp.ndarray of shape (N,), values in {0, 1}.
                        1 = incorrect response, 0 = correct response.
        margin:         Minimum required energy gap between incorrect and correct.
                        Default 1.0 (matches CPMI baseline from Exp 593).
        lambda_calib:   Weight of ECE calibration term.  Default 0.1 — calibration
                        regularises but does not dominate the contrastive signal.

    Returns:
        Scalar float: L_contrastive + lambda_calib * L_calibration.
        Returns 0.0 for empty or single-class batches.

    Spec: REQ-VERIFY-120, REQ-VERIFY-121,
          SCENARIO-VERIFY-157, SCENARIO-VERIFY-158, SCENARIO-VERIFY-159
    """
    if energy_scores.shape[0] == 0:
        return jnp.array(0.0)

    labels_float = labels.astype(jnp.float32)
    # correct_w[i] = 1.0 if label[i]=0 (correct), 0.0 otherwise.
    # incorrect_w[j] = 1.0 if label[j]=1 (incorrect), 0.0 otherwise.
    correct_w = 1.0 - labels_float        # shape (N,)
    incorrect_w = labels_float            # shape (N,)

    # --- Contrastive margin loss (hinge over all correct/incorrect pairs) ----
    # Vectorised outer product to avoid dynamic boolean indexing (which would
    # create dynamic shapes — disallowed inside jax.jit/jax.grad).
    # gaps[i, j] = energy_scores[j] - energy_scores[i]
    #   where i=correct, j=incorrect is weighted by pair_w[i, j].
    gaps = energy_scores[None, :] - energy_scores[:, None]      # (N, N)
    pair_w = correct_w[:, None] * incorrect_w[None, :]          # (N, N)
    pair_losses = jnp.maximum(jnp.array(0.0), margin - gaps) * pair_w
    n_pairs = jnp.sum(pair_w)
    l_contrastive = jnp.where(n_pairs > 0, jnp.sum(pair_losses) / n_pairs, 0.0)

    # --- Calibration loss (ECE) -------------------------------------------
    # sigmoid(energy) = P(incorrect): probability the response is wrong.
    # Convention matches labels: label=1 means incorrect, so P(label=1) = sigmoid(energy).
    predicted_probs = jax.nn.sigmoid(energy_scores)
    l_calibration = ece_loss(predicted_probs, labels_float)

    return l_contrastive + lambda_calib * l_calibration
