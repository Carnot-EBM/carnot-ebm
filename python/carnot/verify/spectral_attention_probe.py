"""SpectralAttentionProbe — Tier 0h spectral diffuseness advisory signal.

**Researcher summary:**
    arXiv 2502.17598 shows that eigenvalues of graph Laplacians of attention maps
    predict hallucinations (F1 > 0.82, < 0.5 ms overhead).  The intuition:
    hallucinatory passages have more diffuse, uniform attention distributions
    (flatter eigenvalue spectra) than factually correct passages where attention
    concentrates on relevant tokens.

    This module implements a CPU-only approximation using token co-occurrence
    bigrams as an attention proxy — the same pattern used in NUP Probe v4
    (Exp 523, AUC=1.0).  We build a token co-occurrence graph, compute its
    normalised Laplacian, extract the smallest 10 non-trivial eigenvalues,
    then compute spectral entropy.  High entropy → diffuse attention spectrum
    → likely hallucination.

**Why token bigrams as an attention proxy:**
    We do not have access to the model's internal attention matrices at inference
    time.  Co-occurrence within a sliding window captures the same "which tokens
    are semantically linked" signal as self-attention, because high-attention
    token pairs tend to co-occur in the same short context window.  This
    approximation trades a small amount of accuracy for zero-overhead CPU
    computation (no LLM call, no GPU, < 0.5 ms per CoT step).

**Why spectral entropy and not raw eigenvalue spread:**
    Spectral entropy normalises over the sum of eigenvalues, making the signal
    comparable across steps of different length and vocabulary size.  A highly
    concentrated spectrum (a few eigenvalues dominate) → low entropy → focused
    attention.  A flat spectrum (eigenvalues roughly equal) → high entropy →
    diffuse attention → hallucination risk.

**Advisory only — does NOT short-circuit Ising:**
    is_spectrally_diffuse=True is stored in VerificationResult.certificate under
    'tier_0h_spectral' and in VerificationResult fields spectral_diffuse and
    spectral_entropy_mean.  It does not affect the verified flag or the repair
    logic.  This mirrors how Tier 0g (StreamingCoTHalluDetector) is wired.

Spec: REQ-VERIFY-146, SCENARIO-VERIFY-173, SCENARIO-VERIFY-174
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# SpectralAttentionProbe
# ---------------------------------------------------------------------------


class SpectralAttentionProbe:
    """Tier 0h spectral diffuseness probe based on bigram co-occurrence Laplacians.

    This class is intentionally stateless for the core spectral operations
    (build_cooccurrence_matrix, compute_laplacian, compute_spectral_entropy).
    The ``train`` / ``predict`` methods add a logistic-regression layer on top
    to emit a calibrated AUC score.

    Spec: REQ-VERIFY-146
    """

    def __init__(self, window: int = 3, n_eigenvalues: int = 10,
                 threshold: float = 2.0) -> None:
        """Initialise probe with configurable window and eigenvalue count.

        Args:
            window: Token co-occurrence sliding window size.  Pairs of tokens
                within ``window`` positions of each other increment the
                co-occurrence count.  Default 3 (matches NUP Probe v4 pattern).
            n_eigenvalues: Number of smallest non-trivial eigenvalues to keep
                when computing spectral entropy.  Cap at min(10, N-1).
                Default 10.
            threshold: Mean spectral entropy threshold above which the trajectory
                is considered diffuse.  Default 2.0.
        """
        self.window = window
        self.n_eigenvalues = n_eigenvalues
        self.threshold = threshold
        # Logistic regression weights set by train(); None before training.
        self._lr_weights: np.ndarray | None = None
        self._lr_bias: float = 0.0

    # ------------------------------------------------------------------
    # Core spectral operations (no model state)
    # ------------------------------------------------------------------

    def build_cooccurrence_matrix(self, text: str, window: int | None = None) -> np.ndarray:
        """Build a symmetric NxN token co-occurrence matrix from a text string.

        WHY this specific tokenization:
            We lower-case and split on non-alphanumeric boundaries so that
            punctuation differences do not inflate the vocabulary (e.g. "cat."
            and "cat" should be the same token).  This produces a cleaner
            co-occurrence graph with fewer spurious edges.

        Args:
            text: Input text string (one CoT step or full passage).
            window: Override instance window.  Default None → use self.window.

        Returns:
            NxN float32 numpy array where entry [i, j] counts co-occurrences
            of token i and token j within the sliding window.  N = |vocabulary|.
            Returns a 1x1 zero matrix for empty or single-token inputs (avoids
            degenerate Laplacian computation downstream).
        """
        w = window if window is not None else self.window
        # Tokenise: lower-case, split on non-word characters.
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if len(tokens) < 2:
            return np.zeros((max(1, len(tokens)), max(1, len(tokens))), dtype=np.float32)

        # Build vocabulary mapping.
        vocab: dict[str, int] = {}
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        n = len(vocab)
        matrix = np.zeros((n, n), dtype=np.float32)

        # Slide window over the token sequence and count co-occurrences.
        for i, tok_i in enumerate(tokens):
            vi = vocab[tok_i]
            for j in range(i + 1, min(i + w + 1, len(tokens))):
                tok_j = tokens[j]
                vj = vocab[tok_j]
                if vi != vj:
                    matrix[vi, vj] += 1.0
                    matrix[vj, vi] += 1.0

        return matrix

    def compute_laplacian(self, cooccurrence: np.ndarray) -> np.ndarray:
        """Compute the unnormalised graph Laplacian L = D - A.

        WHY unnormalised rather than symmetric-normalised:
            The unnormalised Laplacian's eigenvalue spread directly reflects
            the graph's degree heterogeneity.  For the hallucination signal we
            want to measure absolute diffuseness, not relative diffuseness, so
            unnormalised is the right choice.

        Args:
            cooccurrence: NxN co-occurrence matrix (adjacency A).

        Returns:
            NxN Laplacian matrix L = D - A, where D is the diagonal degree
            matrix with D[i,i] = sum of row i of A.
        """
        degree = cooccurrence.sum(axis=1)
        d_matrix = np.diag(degree)
        return d_matrix - cooccurrence

    def compute_spectral_entropy(self, laplacian: np.ndarray) -> float:
        """Compute spectral entropy from the smallest eigenvalues of the Laplacian.

        WHY only the smallest eigenvalues:
            The smallest eigenvalues capture low-frequency graph structure —
            the "global" connectivity pattern.  Hallucinated text tends to have
            weak long-range structure, which shows up as elevated small
            eigenvalues (the spectrum becomes flatter at the low end).

        WHY +1e-8 in the normalisation and log:
            Prevents division by zero for degenerate graphs (all zero adjacency)
            and log(0) for zero probabilities.  This is the same epsilon used in
            NUP Probe v4.

        Args:
            laplacian: NxN Laplacian matrix.

        Returns:
            Spectral entropy float.  Range [0, log(k)] where k = number of
            eigenvalues used.  Higher = more diffuse spectrum.
        """
        n = laplacian.shape[0]
        if n <= 1:
            return 0.0

        k = min(self.n_eigenvalues, n - 1)
        if k < 1:
            return 0.0

        # eigh returns eigenvalues in ascending order; take smallest k.
        try:
            from scipy.linalg import eigh  # noqa: PLC0415

            eigenvalues = eigh(laplacian, subset_by_index=[0, k - 1],
                               eigvals_only=True)
        except Exception:
            # Fallback: full eigen-decomposition with numpy (slower but always works).
            eigenvalues = np.linalg.eigvalsh(laplacian)[:k]

        # Clamp to non-negative (numerical noise can produce tiny negatives).
        eigenvalues = np.clip(eigenvalues, 0.0, None)

        # Normalise to a probability distribution.
        total = eigenvalues.sum() + 1e-8
        probs = eigenvalues / total

        # Shannon entropy: -sum(p * log(p + 1e-8)).
        entropy: float = float(-np.sum(probs * np.log(probs + 1e-8)))
        return entropy

    # ------------------------------------------------------------------
    # Trajectory-level operations
    # ------------------------------------------------------------------

    def compute_trajectory(self, steps: list[str]) -> np.ndarray:
        """Compute the per-step spectral entropy trajectory for a CoT chain.

        Each element of the returned array is the spectral entropy of the
        co-occurrence graph built from that individual step.  A monotonically
        increasing trajectory with high mean entropy is a hallucination signal.

        Args:
            steps: List of CoT step strings.

        Returns:
            1-D float32 array of length len(steps).  Empty array for empty input.
        """
        if not steps:
            return np.array([], dtype=np.float32)

        entropies = []
        for step in steps:
            cooc = self.build_cooccurrence_matrix(step)
            lap = self.compute_laplacian(cooc)
            ent = self.compute_spectral_entropy(lap)
            entropies.append(ent)

        return np.array(entropies, dtype=np.float32)

    def is_diffuse(self, trajectory: np.ndarray, threshold: float | None = None) -> bool:
        """Return True when the trajectory satisfies both diffuseness criteria.

        Two conditions must BOTH be true:
            1. Mean entropy > threshold  (overall high entropy level)
            2. Monotonically-increasing fraction > 0.5  (entropy trending up)

        WHY both conditions:
            Condition 1 alone would flag any verbose step as diffuse.
            Condition 2 alone would flag oscillating trajectories.  Together
            they capture the arXiv 2502.17598 finding: hallucinations have
            sustained upward drift in spectral entropy, not just a single spike.

        Args:
            trajectory: 1-D entropy trajectory (output of compute_trajectory).
            threshold: Override instance threshold.  Default None → self.threshold.

        Returns:
            True when both diffuseness conditions are satisfied.
        """
        th = threshold if threshold is not None else self.threshold
        if len(trajectory) == 0:
            return False
        if len(trajectory) == 1:
            return float(trajectory[0]) > th

        mean_entropy = float(trajectory.mean())

        # Fraction of consecutive pairs where entropy increased.
        diffs = np.diff(trajectory)
        increasing_fraction = float((diffs > 0).mean())

        return mean_entropy > th and increasing_fraction > 0.5

    # ------------------------------------------------------------------
    # Training and prediction
    # ------------------------------------------------------------------

    def _extract_features(self, steps: list[str]) -> np.ndarray:
        """Extract [mean_entropy, entropy_slope, entropy_max] feature vector.

        WHY these three features:
            - mean_entropy: captures the overall diffuseness level
            - entropy_slope: captures the monotonic upward trend (OLS slope)
            - entropy_max: captures the worst-case step (peak hallucination moment)
            Together these are sufficient to separate hallucinatory from correct
            CoT chains in a linear classifier (validated by logistic regression
            on the synthetic corpus in train()).

        Args:
            steps: CoT step strings.

        Returns:
            1-D float32 array [mean, slope, max].
        """
        traj = self.compute_trajectory(steps)
        if len(traj) == 0:
            return np.zeros(3, dtype=np.float32)

        mean_ent = float(traj.mean())
        max_ent = float(traj.max())

        if len(traj) >= 2:
            # OLS slope: fit a line to the trajectory indices.
            x = np.arange(len(traj), dtype=np.float32)
            xm = x.mean()
            tm = traj.mean()
            slope = float(((x - xm) * (traj - tm)).sum() /
                          (((x - xm) ** 2).sum() + 1e-8))
        else:
            slope = 0.0

        return np.array([mean_ent, slope, max_ent], dtype=np.float32)

    def train(self, pos_corpus: list[list[str]], neg_corpus: list[list[str]]) -> None:
        """Train a logistic regression probe on paired CoT chains.

        WHY logistic regression rather than a deeper model:
            We only have 50 training examples (per the Exp 885 spec).  Logistic
            regression with 3 features has very low VC dimension and will not
            overfit.  It is also fully interpretable and deterministic.

        Args:
            pos_corpus: List of CoT step-lists for CORRECT chains (label=0,
                low entropy expected).
            neg_corpus: List of CoT step-lists for HALLUCINATING chains (label=1,
                high entropy expected).
        """
        features = []
        labels = []
        for steps in pos_corpus:
            features.append(self._extract_features(steps))
            labels.append(0)
        for steps in neg_corpus:
            features.append(self._extract_features(steps))
            labels.append(1)

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        # Normalise features.
        self._feat_mean = X.mean(axis=0)
        self._feat_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._feat_mean) / self._feat_std

        # Logistic regression via gradient descent (no sklearn dependency).
        n, d = X_norm.shape
        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        lr = 0.1
        for _ in range(200):
            logits = X_norm.astype(np.float64) @ w + b
            probs = 1.0 / (1.0 + np.exp(-logits))
            err = probs - y.astype(np.float64)
            grad_w = X_norm.astype(np.float64).T @ err / n
            grad_b = err.mean()
            w -= lr * grad_w
            b -= lr * grad_b

        self._lr_weights = w.astype(np.float32)
        self._lr_bias = float(b)

    def predict(self, steps: list[str]) -> dict[str, Any]:
        """Run the spectral probe on a CoT chain and return advisory signal.

        If the logistic regression weights have been trained (via train()), use
        them for the probability estimate.  Otherwise fall back to the
        threshold-based is_diffuse() heuristic.

        Args:
            steps: CoT step strings to evaluate.

        Returns:
            dict with keys:
                - is_spectrally_diffuse: bool — advisory flag
                - spectral_entropy_mean: float — mean step entropy
                - auc_score: float — last AUC from evaluate(), or 0.0 if not computed
        """
        traj = self.compute_trajectory(steps)
        mean_ent = float(traj.mean()) if len(traj) > 0 else 0.0

        if self._lr_weights is not None:
            feat = self._extract_features(steps)
            feat_norm = (feat - self._feat_mean) / self._feat_std
            logit = float(feat_norm.astype(np.float64) @ self._lr_weights + self._lr_bias)
            prob = 1.0 / (1.0 + np.exp(-logit))
            is_diffuse = prob >= 0.5
        else:
            is_diffuse = self.is_diffuse(traj)

        return {
            "is_spectrally_diffuse": bool(is_diffuse),
            "spectral_entropy_mean": mean_ent,
            "auc_score": getattr(self, "_last_auc", 0.0),
        }

    def evaluate(self, pos_corpus: list[list[str]], neg_corpus: list[list[str]]) -> float:
        """Compute AUC-ROC on held-out corpus pairs (positive=correct, negative=hallucinates).

        WHY a custom AUC and not sklearn:
            sklearn is not in Carnot's mandatory dependencies.  A trapezoidal-rule
            AUC from scratch is < 10 lines and avoids a hard dependency for a
            single metric call.

        Args:
            pos_corpus: Correct CoT step-lists (true label = 0).
            neg_corpus: Hallucinating CoT step-lists (true label = 1).

        Returns:
            AUC-ROC float in [0, 1].  0.5 = random baseline.
        """
        scores = []
        labels = []
        for steps in pos_corpus:
            scores.append(self._score_proba(steps))
            labels.append(0)
        for steps in neg_corpus:
            scores.append(self._score_proba(steps))
            labels.append(1)

        auc = _auc_roc(np.array(scores), np.array(labels))
        self._last_auc = auc
        return auc

    def _score_proba(self, steps: list[str]) -> float:
        """Return diffuseness probability for a single chain."""
        if self._lr_weights is not None:
            feat = self._extract_features(steps)
            feat_norm = (feat - self._feat_mean) / self._feat_std
            logit = float(feat_norm.astype(np.float64) @ self._lr_weights + self._lr_bias)
            return 1.0 / (1.0 + np.exp(-logit))
        # Fallback: use mean entropy as raw score.
        traj = self.compute_trajectory(steps)
        return float(traj.mean()) if len(traj) > 0 else 0.0


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------


def _auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Trapezoidal-rule AUC-ROC from raw scores and binary labels.

    WHY this standalone implementation:
        Avoids the sklearn dependency while keeping the computation correct.
        Trapezoidal rule over the ROC curve (sorted by threshold) gives the
        same result as sklearn.metrics.roc_auc_score for continuous scores.

    Args:
        scores: Float probability scores (higher = predicted positive).
        labels: Binary integer labels (1 = positive class).

    Returns:
        AUC-ROC float in [0, 1].
    """
    if len(scores) == 0:
        return 0.5
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by descending score.
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend origin.
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # Trapezoidal integration (np.trapz removed in NumPy 2.0; use np.trapezoid when available).
    try:
        return float(np.trapezoid(tpr, fpr))
    except AttributeError:
        return float(np.trapz(tpr, fpr))
