"""PPSConstraintSelector — freeze low-variance coupling weights during self-play adaptation.

**Researcher summary (PPSEBM — arXiv 2512.15658):**
    Progressive Parameter Selection with Energy-Based Models (PPSEBM) prevents
    catastrophic forgetting during continual learning by identifying which model
    parameters are already task-relevant (low energy variance) and freezing them
    so gradient updates cannot overwrite their learned values.

    Applied to Carnot: during self-play, the VerifyRepairPipeline's coupling weights
    J_(ij) can be destabilized by high-learning-rate updates on new question types,
    erasing the discrimination learned during CD pre-training (the "relapse" pattern
    seen in Exps 697 and 737 where fp_rate rose again after initial improvement).

    The fix (this module):
        1. Track the energy contribution of each coupling over a rolling window of W=30
           questions.  The contribution of coupling (i,j) to the energy at question t
           is J_(ij) * s_i * s_j, which fluctuates as the spin state changes.
        2. Couplings whose contribution has settled (variance < FREEZE_THRESHOLD) are
           "frozen": their gradient component is zeroed before any weight update.
        3. Frozen couplings cannot be overwritten by self-play gradient steps even if
           those steps would otherwise push them in the wrong direction.

**Why energy variance as the freezing criterion:**
    A coupling J_(ij) that consistently has the same contribution s_i * s_j across
    many questions has found a stable discriminative role — it always fires (or always
    doesn't fire) regardless of question content.  Perturbing it would only add noise.
    High variance means the coupling is still exploring its role and should keep learning.

**Relationship to SRSA (Exp 756):**
    SRSA gates BAD data from entering memory (prevents poisoning at the input).
    PPS gates gradient updates FROM overwriting good coupling weights (prevents
    catastrophic forgetting at the parameter level).  Together they close both
    failure modes: bad data in, and good weights overwritten.

Spec: REQ-LEARN-042, SCENARIO-LEARN-082
"""

from __future__ import annotations

from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# CouplingVarianceTracker
# ---------------------------------------------------------------------------


class CouplingVarianceTracker:
    """Rolling variance tracker for energy contributions of each coupling weight.

    **What "energy contribution" means:**
        For an Ising model the energy of state s under coupling matrix J is:
            E(s) = -sum_{i<j} J_{ij} s_i s_j  (up to biases and beta)
        The contribution of coupling (i,j) to that energy is J_{ij} * s_i * s_j.
        This value changes every question because the spin state s changes as the
        pipeline verifies a new (question, response) pair.

        We flatten the N×N coupling matrix into an N²-element vector for bookkeeping.
        In practice the caller passes the per-coupling product J_{ij}*s_i*s_j as
        ``coupling_contributions[k]`` for each coupling index k.

    **Rolling window:**
        We keep only the last ``window_size`` (default 30) observations per coupling.
        Older observations are evicted from the left of the deque automatically.
        This lets the variance adapt as new question types appear without accumulating
        an ever-growing history.

    Args:
        n_couplings:  Number of coupling weights to track.  Equals n_spins² for a
                      fully connected Ising model, or the number of non-zero edges
                      for a sparse one.
        window_size:  Rolling window length.  30 matches the W=30 value from the
                      PPSEBM paper and from the self-play window used in REQ-PSV-015.

    Spec: REQ-LEARN-042
    """

    def __init__(self, n_couplings: int, window_size: int = 30) -> None:
        self.n_couplings = n_couplings
        self.window_size = window_size
        # One deque per coupling, each holding the last window_size contributions.
        self._windows: list[deque[float]] = [
            deque(maxlen=window_size) for _ in range(n_couplings)
        ]

    def update(self, coupling_contributions: np.ndarray) -> None:
        """Record the energy contribution of each coupling for one question.

        **Why one update per question (not per gradient step):**
            The contribution J_{ij}*s_i*s_j reflects the current spin state, which
            is determined by the verification pass on the current question.  Tracking
            per-question gives us a question-distribution view of how stable each
            coupling's role is — exactly what PPSEBM measures as "task relevance".

        Args:
            coupling_contributions: 1-D array of shape (n_couplings,).
                Element k is the scalar contribution of coupling k to the energy
                for the current question's spin configuration.

        Spec: REQ-LEARN-042
        """
        for k, val in enumerate(coupling_contributions):
            self._windows[k].append(float(val))

    def get_variance(self) -> np.ndarray:
        """Return rolling variance of each coupling's contribution.

        **How variance is computed:**
            For each coupling k we compute the sample variance of the values in its
            window.  If the window has fewer than 2 observations the variance is
            defined as 0.0 — a coupling that has barely been seen is treated as
            stable (conservative: don't update things we can't measure yet).

            We use population variance (divide by N, not N-1) to keep the
            implementation simple and consistent with the PPSEBM paper's formulation.

        Returns:
            1-D float64 array of shape (n_couplings,) with the variance for each
            coupling over the current rolling window.

        Spec: REQ-LEARN-042
        """
        variances = np.zeros(self.n_couplings, dtype=np.float64)
        for k, window in enumerate(self._windows):
            vals = list(window)
            if len(vals) < 2:
                variances[k] = 0.0
                continue
            arr = np.array(vals, dtype=np.float64)
            variances[k] = np.var(arr)  # population variance
        return variances

    def get_frozen_mask(self, freeze_threshold: float = 0.01) -> np.ndarray:
        """Return a boolean mask indicating which couplings are frozen.

        **Semantics:**
            ``True`` at index k means coupling k is frozen — its variance is below
            ``freeze_threshold`` and it MUST NOT be updated during self-play
            adaptation (REQ-LEARN-042).

            ``False`` at index k means coupling k is still learning — its variance
            is at or above threshold and gradient updates are allowed.

        Args:
            freeze_threshold: Variance below this value triggers freezing.
                Default 0.01 comes from the PPSEBM paper's threshold in the
                continual-learning regime and matches SelfLearningRelay.FREEZE_THRESHOLD.

        Returns:
            Boolean array of shape (n_couplings,).  True = frozen.

        Spec: REQ-LEARN-042
        """
        return self.get_variance() < freeze_threshold


# ---------------------------------------------------------------------------
# PPSConstraintSelector
# ---------------------------------------------------------------------------


class PPSConstraintSelector:
    """Apply the PPS frozen-coupling mask to zero out gradient entries.

    **What this does:**
        Before any coupling-weight update step in self-play, the gradient vector
        (one element per coupling) is passed through ``apply_mask``.  Elements
        corresponding to frozen couplings are set to zero, so the update step
        will not change those weights.

    **Why zero the gradient (not skip the update entirely):**
        The optimizer may accumulate state (momentum, Adam second moment) that
        builds up even for entries with zero gradient.  This is acceptable here
        because our self-play adaptation is first-order (SGD-like), and zeroing
        the gradient is the standard PPSEBM approach.  If optimizer state becomes
        an issue, the caller can additionally zero the optimizer state for frozen
        entries.

    Args:
        tracker:          CouplingVarianceTracker with the current variance estimates.
        freeze_threshold: Variance below this → frozen.  Passed through to
                          ``tracker.get_frozen_mask()`` on each call.

    Spec: REQ-LEARN-042
    """

    def __init__(
        self,
        tracker: CouplingVarianceTracker,
        freeze_threshold: float = 0.01,
    ) -> None:
        self._tracker = tracker
        self._freeze_threshold = freeze_threshold

    def apply_mask(self, gradient: np.ndarray) -> np.ndarray:
        """Zero out gradient entries for frozen couplings.

        **What "frozen coupling gradient" means:**
            For coupling k that is frozen (variance < threshold), the k-th element
            of the returned gradient array is exactly 0.0.  Downstream optimizer
            code that computes ``weight[k] -= lr * gradient[k]`` will therefore
            leave weight[k] unchanged for frozen k.

        Args:
            gradient: 1-D array of shape (n_couplings,) containing the raw
                      gradient of the loss w.r.t. each coupling weight.

        Returns:
            A new array of the same shape and dtype with frozen entries zeroed.

        Spec: REQ-LEARN-042
        """
        mask = self._tracker.get_frozen_mask(self._freeze_threshold)
        masked = gradient.copy()
        masked[mask] = 0.0
        return masked

    def frozen_count(self) -> int:
        """Return the number of coupling weights currently frozen.

        Useful for monitoring and for the experiment artifact field
        ``n_frozen_at_step30``.

        Spec: REQ-LEARN-042
        """
        return int(np.sum(self._tracker.get_frozen_mask(self._freeze_threshold)))


__all__ = [
    "CouplingVarianceTracker",
    "PPSConstraintSelector",
]
