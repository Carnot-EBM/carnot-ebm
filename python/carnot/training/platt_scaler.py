"""platt_scaler — Post-hoc temperature (Platt) scaling for JEPA calibration.

**Why temperature scaling exists:**
    JEPA v14 (Exp 631) achieved excellent OOD AUC=0.912 but ECE=0.132, which is above
    the calibration target of 0.10.  The model discriminates correctly but its
    *confidence scores* are systematically overconfident — it claims probabilities
    closer to 0 or 1 than the actual accuracy of those predictions warrants.

    Temperature scaling is the simplest post-hoc calibration method (Guo et al., 2017):
    divide all logits by a single learned scalar T before passing through sigmoid.
    T > 1 softens probabilities (reduces overconfidence).
    T < 1 sharpens probabilities (increases confidence).
    T = 1 leaves the model unchanged.

    The key insight: temperature scaling preserves the *ordering* of predictions
    (AUC does not change), it only rescales the magnitudes.  This is why it is safe
    to apply post-hoc without touching the trained JEPA weights.

**Why post-hoc instead of training-time ECE loss?**
    CAPO loss (Exp 618) applied ECE during training.  v14 still has ECE=0.132 because
    the training corpus has limited diversity — the model learned calibrated predictions
    for its training distribution but remains slightly overconfident on held-out data.
    Temperature scaling corrects this residual bias with a single scalar fitted on a
    calibration split (20% of the corpus), requiring no re-training.

**Reference:**
    Guo et al. "On Calibration of Modern Neural Networks", ICML 2017.
    arXiv 2604.12632 (CAPO, the JEPA training objective).

Spec: REQ-VERIFY-144, SCENARIO-VERIFY-190, SCENARIO-VERIFY-191
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


class PlattScaler:
    """Temperature scaling calibrator for binary classifier logits.

    Temperature scaling applies a single parameter T to raw model logits before
    the sigmoid activation.  Calibrated probability = sigmoid(logit / T).

    Fitting minimises negative log-likelihood on a held-out calibration set using
    simple gradient descent.  T is clipped to [0.1, 10.0] at each step to avoid
    degenerate collapse to 0 or infinity (REQ-VERIFY-144-6).

    Typical usage::

        scaler = PlattScaler()
        T_optimal = scaler.fit(cal_logits, cal_labels)
        cal_probs = scaler.calibrate(val_logits)
        ece = scaler.compute_ece(cal_probs, val_labels)
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        """Initialise with starting temperature.

        Args:
            init_temperature: Starting value for T.  1.0 = no change.  Larger
                              values pre-soften the distribution for faster convergence
                              when the model is known to be overconfident.

        Spec: REQ-VERIFY-144-1
        """
        self.T = float(init_temperature)

    def fit(
        self,
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        n_steps: int = 200,
        lr: float = 0.01,
    ) -> float:
        """Optimise temperature T to minimise NLL on the calibration set.

        Uses vanilla gradient descent — JAX's grad() differentiates the NLL
        loss with respect to T.  The loop is pure Python so T remains a concrete
        scalar that can be clipped without tracing issues.

        **Why NLL and not ECE directly?**
            NLL is smooth and fully differentiable in T, making it a reliable
            proxy for calibration.  Direct ECE minimisation via gradient descent
            requires differentiable bin assignment, which introduces approximation
            error.  NLL produces nearly identical final ECE with less numerical
            instability (Guo et al., 2017 Table 1).

        Args:
            logits: jnp.ndarray of shape (N,) — raw model outputs (pre-sigmoid).
            labels: jnp.ndarray of shape (N,) — 1.0 if incorrect, 0.0 if correct.
            n_steps: Number of gradient descent steps.  200 is sufficient for
                     convergence on typical JEPA calibration sets (validated Exp 646).
            lr:      Learning rate.  0.01 is conservative — converges in ~100 steps
                     without overshooting the optimal T.

        Returns:
            Optimal T (float).  Also stored in self.T for subsequent calibrate() calls.

        Spec: REQ-VERIFY-144-2
        """
        T = float(self.T)
        logits_arr = jnp.array(logits, dtype=jnp.float32)
        labels_arr = jnp.array(labels, dtype=jnp.float32)

        def _nll(t: float) -> jnp.ndarray:
            # Negative log-likelihood of Bernoulli(sigmoid(logit/t)).
            calibrated = jax.nn.sigmoid(logits_arr / t)
            return -jnp.mean(
                labels_arr * jnp.log(calibrated + 1e-8)
                + (1.0 - labels_arr) * jnp.log(1.0 - calibrated + 1e-8)
            )

        grad_nll = jax.grad(_nll)

        for _ in range(n_steps):
            g = float(grad_nll(T))
            T = T - lr * g
            # Clip to prevent degenerate scaling (REQ-VERIFY-144-6).
            T = float(jnp.clip(T, 0.1, 10.0))

        self.T = T
        return T

    def calibrate(self, logits: jnp.ndarray) -> jnp.ndarray:
        """Apply temperature scaling and return calibrated probabilities.

        Divides logits by self.T (fitted or default=1.0) then applies sigmoid.
        When T > 1, probabilities are pulled toward 0.5 (less overconfident).
        When T < 1, probabilities are sharpened (rarely needed after fitting).

        Args:
            logits: jnp.ndarray of shape (N,) — raw model logits.

        Returns:
            jnp.ndarray of shape (N,) with values in (0, 1).

        Spec: REQ-VERIFY-144-3
        """
        return jax.nn.sigmoid(jnp.array(logits, dtype=jnp.float32) / self.T)

    def compute_ece(
        self,
        probs: jnp.ndarray,
        labels: jnp.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error over equal-width probability bins.

        **Layman explanation:**
            Split all predictions into n_bins buckets from 0% to 100% confidence.
            For each bucket, compare the average predicted confidence against the
            fraction of examples in that bucket that were actually correct.
            ECE = weighted average of those gaps (weight = fraction of total samples).
            ECE=0 means perfect calibration; ECE=0.10 means a 10% average gap.

        This is the standard Guo et al. (2017) weighted ECE — each bin contributes
        proportional to its sample count, unlike the mean-over-bins version in
        ece_loss() which treats all non-empty bins equally.

        Args:
            probs:  jnp.ndarray of shape (N,), values in [0, 1].
                    P(response is incorrect) — higher = more likely wrong.
            labels: jnp.ndarray of shape (N,), values in {0, 1}.
                    1 = actually incorrect, 0 = actually correct.
            n_bins: Number of equal-width bins.  Default 10.

        Returns:
            ECE as a Python float.

        Spec: REQ-VERIFY-144-4
        """
        probs_np = np.array(probs, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.float32)
        n = len(probs_np)
        if n == 0:
            return 0.0

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo = bin_boundaries[i]
            hi = bin_boundaries[i + 1]
            # Include the right edge in the last bin to capture prob=1.0 exactly.
            if i < n_bins - 1:
                in_bin = (probs_np >= lo) & (probs_np < hi)
            else:
                in_bin = (probs_np >= lo) & (probs_np <= hi)
            bin_count = int(in_bin.sum())
            if bin_count == 0:
                continue
            bin_acc = float(labels_np[in_bin].mean())
            bin_conf = float(probs_np[in_bin].mean())
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
        return float(ece)
