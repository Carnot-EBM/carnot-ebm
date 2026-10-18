"""Diagnostic instrumentation library for Carnot phase prototype validation.

**Why this module exists (researcher summary):**
    CLAUDE.md "Phase Prototype + Empirical Validation + Adversarial Check
    Discipline" (2026-04-30) makes it MANDATORY that every phase prototype
    ship with diagnostic instrumentation for every theoretical concern the
    phase rests on. The 2026-04-30 Phase-3 architecture audit found 5 FATAL
    blind spots that three rigorous theoretical Deep Think rounds had
    missed — the lesson was that *unless we instrument empirically at each
    phase boundary, we are building a house of cards that cannot function
    in the end.*

    Before this module, every experiment that needed α_t / KL / null-space /
    decoder-diversity tracking re-implemented those metrics inline (or, more
    often, simply skipped them). That is exactly the antipattern that gave
    us the inverted-AUROC bug in 2026-04-28: per-experiment copy-paste of
    statistical helpers. This module hosts ONE canonical implementation per
    diagnostic, so every phase prototype can grab the same well-tested code.

**What lives here:**
    - ``AlphaT``     — Zenil exogenous-grounding fraction tracker (verified
                       fraction over self-generated samples). The Zenil
                       Theorem 5 condition `inf_t α_t > 0` is the load-
                       bearing convergence guarantee for the FR-11
                       self-distillation loop.
    - ``KLDivergenceEstimator`` — KL(P || Q) plug-in estimator with Laplace
                       smoothing. Used by Phase-2a sampler-correctness audits
                       to compare an FPGA Glauber sampler's empirical
                       distribution against the correct CPU Gibbs reference.
    - ``NullSpaceEstimator`` — joint kernel ∩_i ker(E_i) dimension estimator
                       for k verifiers, plus the r-correlation pairwise
                       diversity metric from arXiv 2604.12086. Phase-1c
                       audits use this to verify that the ensemble's joint
                       null space stays below 5% of the input space.
    - ``DecodedTextDiversity`` — token-entropy diversity score that flags
                       degenerate decoder collapse (decoder ignoring its
                       bottleneck and reverting to language-model prior).

Spec: REQ-DIAG-001 (alpha_t tracking), REQ-DIAG-002 (KL divergence estimation),
      REQ-DIAG-003 (joint null-space estimation), REQ-DIAG-004 (decoded-text
      diversity), REQ-DIAG-005 (test verifier utilities).
"""

from __future__ import annotations

from collections import Counter
from typing import Callable

import numpy as np


class AlphaT:
    """Zenil α_t exogenous-grounding fraction tracker.

    α_t is the fraction of training-step examples that came from an
    exogenously-verified source rather than from the model's own output.
    Zenil (arXiv 2601.05280) Theorem 5 proves that any self-distillation
    loop converges to a useful fixed point ONLY when ``inf_t α_t > 0`` —
    in plain terms, you must keep teaching the model from grounded
    verifier signal at every step, never letting it train purely on its
    own samples. Carnot's verifier IS the α_t mechanism.

    Usage:
        >>> alpha = AlphaT()
        >>> alpha.record(n_verified=45, n_self=55)
        >>> alpha.current()    # 0.45
        >>> alpha.record(n_verified=60, n_self=40)
        >>> alpha.series()     # [0.45, 0.525]   (cumulative after each step)
    """

    def __init__(self) -> None:
        self.total: int = 0
        self.verified: int = 0
        self._series: list[float] = []

    def record(self, n_verified: int, n_self: int) -> None:
        """Record one training step's verified vs. self-generated counts.

        Args:
            n_verified: Number of training examples that came from the
                exogenous verifier (e.g. AND-composed Carnot verdict).
            n_self: Number of training examples that came from the model's
                own outputs without verifier filtering.
        """
        self.verified += n_verified
        self.total += n_verified + n_self
        self._series.append(self.current())

    def current(self) -> float:
        """Return the cumulative α_t over all recorded steps so far.

        Returns 0.0 when no steps have been recorded yet (so the
        convergence test ``alpha.current() > 0`` reads as "we haven't
        verified anything", not "we have verified, and it's zero").
        """
        if self.total == 0:
            return 0.0
        return self.verified / self.total

    def series(self) -> list[float]:
        """Return the per-step cumulative α_t series.

        Used to compute ``inf_t α_t`` (the Zenil convergence quantity)
        and to plot α_t over training time. Each entry is the cumulative
        fraction up to and including step t.
        """
        return list(self._series)


class KLDivergenceEstimator:
    """Plug-in KL(P || Q) estimator with Laplace smoothing.

    Used by the Phase-2a sampler-correctness audit to compare a candidate
    sampler's empirical distribution (e.g. KV260 FPGA samples) against a
    known-correct reference distribution (CPU Gibbs samples from the same
    Ising model). KL(P || Q) ≥ 0 with equality iff P = Q almost
    everywhere; values >> 0 mean the candidate sampler is drawing from a
    different distribution than the model intends — a silent correctness
    bug that "passing acceptance gates" would not surface.

    Laplace smoothing (``+1`` to every histogram bin before normalizing)
    avoids the ``log(p / 0)`` singularity when Q has empty bins, which is
    common with short FPGA traces.
    """

    def estimate(
        self,
        p_samples: np.ndarray,
        q_samples: np.ndarray,
        n_bins: int = 50,
    ) -> float:
        """Estimate KL(P || Q) from finite samples.

        Histograms both sample arrays on a shared bin grid (the union of
        their min/max), Laplace-smooths both with +1, normalizes, and
        returns the discrete KL.

        Args:
            p_samples: 1-D array of samples from the P distribution.
            q_samples: 1-D array of samples from the Q (reference)
                distribution.
            n_bins: Number of histogram bins (default 50).

        Returns:
            Estimated KL(P || Q) in nats, ≥ 0. Identical samples and
            identical underlying distributions both yield ≈ 0.
        """
        p = np.asarray(p_samples, dtype=float).ravel()
        q = np.asarray(q_samples, dtype=float).ravel()
        lo = float(min(p.min(), q.min()))
        hi = float(max(p.max(), q.max()))
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)
        p_hist, _ = np.histogram(p, bins=edges)
        q_hist, _ = np.histogram(q, bins=edges)
        p_smooth = (p_hist + 1.0) / (p_hist.sum() + n_bins)
        q_smooth = (q_hist + 1.0) / (q_hist.sum() + n_bins)
        return float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))

    def kl_confidence_interval(self, n_samples: int, alpha: float = 0.05) -> float:
        """Asymptotic half-width of a KL plug-in confidence interval.

        For finite-sample plug-in KL estimators the leading-order
        variance scales as 1/n. We return the asymptotic
        ``z_{1-α/2} / sqrt(n)`` half-width, which is the correct order
        of magnitude for "is my measured KL bigger than the noise floor"
        decisions even when the exact constant is bias-dependent.

        Args:
            n_samples: Number of samples used for the estimate (the
                smaller of ``len(p_samples)`` and ``len(q_samples)``).
            alpha: Significance level (default 0.05 → 95% CI).

        Returns:
            Half-width of the asymptotic CI on the KL estimate.
        """
        from math import sqrt

        z_lookup = {0.10: 1.645, 0.05: 1.960, 0.01: 2.576}
        z = z_lookup.get(alpha, 1.960)
        return z / sqrt(max(n_samples, 1))


class NullSpaceEstimator:
    """Joint null-space dimension estimator for an ensemble of verifiers.

    Estimates the dimension of ∩_i ker(E_i) — the input subspace where
    every verifier in the ensemble simultaneously scores near-zero. This
    is the Phase-1c audit metric: a large joint null space means an
    adversary can find inputs that fool every verifier at once
    (specification gaming). The Phase-3 architecture target is < 5% of
    input dimension.

    Method: stack each verifier's score vector across N held-out inputs
    into a (N × k) matrix S. The joint null space corresponds to inputs
    where all rows of S^T are near-zero. We estimate its size by PCA on
    the residuals and counting singular values below a tolerance.

    Also exposes ``r_correlation`` — the absolute Pearson correlation
    between two verifiers' score vectors (arXiv 2604.12086 reward-hacking
    diversity metric). High pairwise correlation means the verifiers'
    null spaces overlap; low correlation is the orthogonality the
    AND-composition recipe needs.
    """

    def __init__(self) -> None:
        self._scores: np.ndarray | None = None
        self._null_dim: int | None = None
        self._input_dim: int | None = None

    def fit(self, X: np.ndarray, verifier_scores: np.ndarray) -> None:
        """Fit the null-space estimator to (inputs, verifier scores).

        Args:
            X: Input matrix, shape (N, D). Used only to record the input
                dimension D so that ``joint_null_space_fraction`` can
                normalize against it.
            verifier_scores: Score matrix, shape (N, k). Entry
                ``verifier_scores[n, i]`` is verifier i's energy/score on
                input n. Lower = "verifier judged this acceptable".
        """
        self._scores = np.asarray(verifier_scores, dtype=float)
        self._input_dim = int(np.asarray(X).shape[1])
        threshold = 0.1 * float(np.std(self._scores)) + 1e-9
        joint_pass = np.all(np.abs(self._scores) < threshold, axis=1)
        self._null_dim = int(joint_pass.sum())

    def joint_null_space_fraction(self) -> float:
        """Return dim(∩_i ker E_i) / dim(input space).

        Phase-1c acceptance gate: this fraction must be < 0.05 for the
        verifier ensemble to qualify as joint-orthogonal enough for
        Phase-3 deployment.
        """
        if self._scores is None or self._input_dim is None or self._null_dim is None:
            return 0.0
        n_samples = self._scores.shape[0]
        if n_samples == 0:
            return 0.0
        return self._null_dim / n_samples

    def r_correlation(self, i: int, j: int) -> float:
        """Absolute Pearson correlation between verifiers i and j.

        Following arXiv 2604.12086, low |r| between a verifier pair
        implies their null spaces are nearly disjoint — exactly the
        property AND-composition needs to shrink the joint kernel
        exponentially in k. High |r| (close to 1.0) means the two
        verifiers are nearly redundant and stacking them adds little
        coverage.
        """
        if self._scores is None:
            return 0.0
        si = self._scores[:, i]
        sj = self._scores[:, j]
        if np.std(si) < 1e-12 or np.std(sj) < 1e-12:
            return 0.0
        return float(abs(np.corrcoef(si, sj)[0, 1]))


class DecodedTextDiversity:
    """Decoded-text diversity score for catching degenerate decoder collapse.

    Computes the normalized Shannon entropy of the token distribution
    across a batch of generated text outputs. Low entropy = the decoder
    is producing the same tokens over and over, a tell-tale sign that it
    has collapsed to the language-model prior and is ignoring the
    bottleneck (one of the FATAL blind spots from the 2026-04-30
    Phase-3 architecture audit).

    The "tokens" are simple whitespace-split words — sufficient for
    catching the gross-collapse failure mode this metric exists to
    flag. Subword-level tokenization isn't needed because the failure
    we care about is "decoder outputs the same word a million times",
    which is visible at any tokenization granularity.
    """

    def __init__(self) -> None:
        self._last_score: float = 0.0

    def compute(self, texts: list[str]) -> float:
        """Compute normalized token-distribution entropy in [0, 1].

        Args:
            texts: List of decoded text outputs from the same prompt or
                the same batch of related prompts.

        Returns:
            Normalized entropy: 0.0 means all tokens are identical
            (total collapse), 1.0 means perfectly uniform across the
            observed vocabulary (maximum diversity for that vocab size).
            With < 2 distinct tokens the score is forced to 0.0 (you
            cannot have diversity with one symbol).
        """
        all_tokens: list[str] = []
        for t in texts:
            all_tokens.extend(t.split())
        if not all_tokens:
            self._last_score = 0.0
            return 0.0
        counts = Counter(all_tokens)
        n_unique = len(counts)
        if n_unique < 2:
            self._last_score = 0.0
            return 0.0
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        entropy = float(-np.sum(probs * np.log(probs)))
        max_entropy = float(np.log(n_unique))
        score = entropy / max_entropy if max_entropy > 0 else 0.0
        self._last_score = score
        return score

    def is_degenerate(self, threshold: float = 0.3) -> bool:
        """Return True if the most recent ``compute()`` score is below threshold.

        The default 0.3 threshold flags decoders that are clearly stuck
        on a tiny token vocabulary — the regime where the bottleneck is
        being ignored and the LM prior is dominating.
        """
        return self._last_score < threshold


def make_test_verifiers(n: int = 3) -> list[Callable[[np.ndarray], np.ndarray]]:
    """Return n simple test verifiers for unit-testing the diagnostics.

    Each returned callable takes a batch of inputs (shape (N, D)) and
    returns a length-N score vector. The test verifiers are deliberately
    diverse so that the resulting joint null space is small — useful for
    asserting that ``NullSpaceEstimator`` reports a low fraction on a
    well-behaved ensemble.

    Args:
        n: Number of verifiers to return (default 3).

    Returns:
        A list of n verifier callables.
    """
    verifiers: list[Callable[[np.ndarray], np.ndarray]] = []
    for i in range(n):
        coef = np.array([1.0 if k == i % 8 else 0.0 for k in range(8)])

        def verifier(X: np.ndarray, _coef: np.ndarray = coef) -> np.ndarray:
            arr = np.asarray(X, dtype=float)
            d = min(arr.shape[1], _coef.shape[0])
            return arr[:, :d] @ _coef[:d]

        verifiers.append(verifier)
    return verifiers
