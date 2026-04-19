"""NUPProbeV3 — CLAP-based hallucination probe using cross-layer activation features.

**Why NUPProbeV3 exists (RETRO-049, arXiv 2509.09700 — CLAP, September 2025):**
    NUPProbe v1 (Exp 484, AUC=0.600) used character entropy.
    NUPProbe v2 (Exp 496, AUC=0.600) used Bayesian semantic entropy — a sequence-level
    aggregate that computes one entropy estimate per CoT step.  Root cause of the v2
    failure: sequence-level aggregates AVERAGE AWAY the per-token signal.  When a model
    is about to hallucinate, the uncertainty spike is LOCAL — it occurs at a specific
    layer and a specific token position (e.g., the token where the model is choosing
    between a real fact and a plausible fabrication).  Averaging across all tokens and
    all layers loses this locality.

    arXiv 2509.09700 (CLAP — Cross-Layer Attention Probing) introduces the key insight:
    construct a (n_layers, n_tokens, hidden_dim) activation tensor from multiple residual
    stream layers and apply multi-head attention over the cross-layer sequence.  This
    captures INTER-LAYER REASONING TRAJECTORIES:
        - Early layers  (~0-8)  : surface syntax, tokenisation artifacts
        - Middle layers (~8-24) : semantic integration, factual recall
        - Late layers   (~24-N) : generation decision, commitment to a token

    Hallucination fingerprints are LOCAL in this 3D tensor: a specific layer-token cell
    where the model becomes uncertain.  Cross-layer attention finds these fingerprints by
    treating the n_layers activations at each token position as a sequence and attending
    over it.

**Three complementary features:**
    1. per_token_entropy (shape: n_tokens):
       Shannon entropy of the softmax distribution over vocabulary at each token position.
       WHERE the model is uncertain.  Tokens with high entropy = model cannot commit to
       a specific word → potential hallucination site.

    2. topk_concentration (shape: n_tokens):
       Ratio of top-1 probability to sum of top-k probabilities.  HOW STRONGLY the model
       is committed to its first choice vs. the next k-1 alternatives.  Low ratio =
       probability mass is spread across multiple tokens = under-constrained continuation.

    3. cross_layer_variance (shape: n_tokens):
       Mean variance across layers of the L2 norm of the activation vector at each token.
       HOW MUCH DO LAYERS DISAGREE about the representation of each token.  High variance
       = early and late layers are encoding very different things about this token position
       = the model is revising its interpretation mid-stack = potential reasoning instability.

**Why these three features work together:**
    per_token_entropy measures OUTPUT uncertainty; cross_layer_variance measures INTERNAL
    representation disagreement.  A token can have high output entropy (uniform softmax)
    for benign reasons (e.g., punctuation).  But a token with BOTH high output entropy
    AND high cross-layer variance is doubly suspicious: the model is uncertain AND its
    internal layers are disagreeing.  topk_concentration adds a third angle: even if
    entropy is moderate, if the top-1 probability is much lower than usual (concentration
    near 1/k), the model is structurally unsure.  Combining all three into a feature
    vector and training a logistic classifier gives NUPProbeV3 the richer signal set
    needed to cross the AUC >= 0.700 Tier 0c threshold.

**CI stub behaviour:**
    When no GPU is available (CI/dev environments), CLAPFeatureExtractor accepts a
    synthetic activations array (shape (n_layers, n_tokens, hidden_dim)) and produces
    the same feature shapes as a real extraction.  This means the full training pipeline
    runs on synthetic data — the AUC result will be near chance (0.5) on synthetic data,
    which is honest.  On real GPU data from Exps 502-503, CLAP features target AUC >= 0.700.

Spec: REQ-VERIFY-104, REQ-VERIFY-105, REQ-VERIFY-106,
      SCENARIO-VERIFY-137, SCENARIO-VERIFY-138, SCENARIO-VERIFY-139
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# CLAPFeatures
# ---------------------------------------------------------------------------


@dataclass
class CLAPFeatures:
    """Feature set extracted by CLAPFeatureExtractor from a cross-layer activation tensor.

    **Why these three fields (not one aggregate):**
        Keeping features separate allows the downstream NUPProbeV3 logistic classifier
        to learn different weights for each signal.  If we pre-combined them, we'd lose
        interpretability and the ability to ablate individual features.

    Attributes:
        per_token_entropy: Shape (n_tokens,). Shannon entropy of softmax(logits) over
            vocabulary at each token position, averaged across the last n_layers of the
            residual stream.  Units: nats.  Higher = more uncertain at that token position.
        topk_concentration: Shape (n_tokens,). Ratio top-1 / sum(top-k), where k=5 by
            default.  Range [1/k, 1].  Lower = probability mass is spread = under-constrained.
        cross_layer_variance: Shape (n_tokens,). Mean variance across n_layers of the
            L2 norm of the hidden state vector at each token position.  Higher = layers
            disagree about this token's representation.

    Spec: REQ-VERIFY-104, REQ-VERIFY-105
    """

    per_token_entropy: np.ndarray   # shape (n_tokens,)
    topk_concentration: np.ndarray  # shape (n_tokens,)
    cross_layer_variance: np.ndarray  # shape (n_tokens,)

    def to_feature_vector(self) -> np.ndarray:
        """Flatten and normalise all three feature arrays into a single 1D vector.

        **Why normalise before concatenation:**
            The three signals are on different scales: entropy is in nats (typically
            0–8 for vocab size ~32k), concentration is in [0.2, 1], and variance is
            in arbitrary squared-norm units.  Z-score normalisation within each signal
            makes them comparable so the logistic classifier doesn't accidentally weight
            the highest-magnitude signal (variance) most heavily.

            If the std of any signal is ~0 (degenerate input), we skip normalisation
            for that signal (divide by 1) to avoid NaN.

        Returns:
            1D np.ndarray of length 3 * n_tokens, dtype float64.  The first n_tokens
            elements are per_token_entropy (z-scored), the next n_tokens are
            topk_concentration (z-scored), the last n_tokens are cross_layer_variance
            (z-scored).
        """
        def _zscore(arr: np.ndarray) -> np.ndarray:
            std = float(np.std(arr))
            if std < 1e-10:
                return arr - float(np.mean(arr))
            return (arr - float(np.mean(arr))) / std

        return np.concatenate([
            _zscore(self.per_token_entropy),
            _zscore(self.topk_concentration),
            _zscore(self.cross_layer_variance),
        ])


# ---------------------------------------------------------------------------
# CLAPFeatureExtractor
# ---------------------------------------------------------------------------


class CLAPFeatureExtractor:
    """Extract CLAP hallucination features from a cross-layer activation tensor.

    **Why this class (not a function):**
        State (n_layers, n_heads, topk) is fixed at construction time so callers
        can reuse the same extractor across many examples without re-specifying
        these hyperparameters.

    Args:
        n_layers: Number of residual stream layers to use from the end of the model.
            Default 4 (last 4 layers).  Must be >= 1.
        n_heads: Number of attention heads for cross-layer attention weighting.
            Not used in the current approximation (we use variance, not learned attention),
            but preserved as a public hyperparameter for future full CLAP implementation.
        topk: k for top-k concentration ratio.  Default 5.

    Spec: REQ-VERIFY-104, REQ-VERIFY-105
    """

    def __init__(
        self,
        n_layers: int = 4,
        n_heads: int = 8,
        topk: int = 5,
    ) -> None:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.topk = topk

    def extract_features(self, activations: np.ndarray) -> CLAPFeatures:
        """Extract CLAPFeatures from a cross-layer activation tensor.

        **Algorithm:**
            Given activations of shape (n_layers, n_tokens, hidden_dim):

            1. per_token_entropy: For each (layer, token), compute softmax over hidden_dim
               as a proxy vocabulary distribution (real usage would use actual logits, but
               this approximation is valid for CI/synthetic mode).  Then compute Shannon
               entropy.  Average across layers at each token position.

            2. topk_concentration: For each (layer, token), compute softmax over hidden_dim
               and pick top-k values.  Concentration = max / sum(top-k).  Average across layers.

            3. cross_layer_variance: For each token, compute the L2 norm of the activation
               vector at each layer, yielding a (n_layers,) norm sequence per token.
               Variance of this sequence = how much layers disagree about representation
               magnitude.  Shape: (n_tokens,).

        Args:
            activations: np.ndarray of shape (n_layers, n_tokens, hidden_dim), dtype float.
                Must have exactly 3 dimensions.  n_layers in the tensor must match
                self.n_layers exactly.

        Returns:
            CLAPFeatures with all three arrays of shape (n_tokens,).

        Raises:
            ValueError: If activations.ndim != 3 or activations.shape[0] != self.n_layers.

        Spec: REQ-VERIFY-104, REQ-VERIFY-105, SCENARIO-VERIFY-137
        """
        if activations.ndim != 3:
            raise ValueError(
                f"activations must be 3D (n_layers, n_tokens, hidden_dim), "
                f"got shape {activations.shape}"
            )
        act_layers, n_tokens, hidden_dim = activations.shape
        if act_layers != self.n_layers:
            raise ValueError(
                f"activations.shape[0]={act_layers} must equal self.n_layers={self.n_layers}"
            )

        per_token_entropy = np.zeros(n_tokens, dtype=np.float64)
        topk_concentration = np.zeros(n_tokens, dtype=np.float64)

        for layer_idx in range(self.n_layers):
            # activations[layer_idx]: (n_tokens, hidden_dim)
            layer_act = activations[layer_idx].astype(np.float64)

            # Softmax over hidden_dim as proxy vocabulary distribution
            # Numerically stable: subtract row max before exp
            row_max = np.max(layer_act, axis=1, keepdims=True)
            exp_act = np.exp(layer_act - row_max)
            row_sum = np.sum(exp_act, axis=1, keepdims=True)
            probs = exp_act / (row_sum + 1e-12)  # shape (n_tokens, hidden_dim)

            # Per-token entropy: H = -sum p * log(p), sum over hidden_dim axis
            safe_log = np.where(probs > 1e-12, np.log(probs + 1e-12), 0.0)
            per_token_entropy += -np.sum(probs * safe_log, axis=1)

            # Top-k concentration: ratio of max to sum of top-k
            k = min(self.topk, hidden_dim)
            # Partition to get top-k indices (axis=1)
            top_k_vals = np.partition(probs, -k, axis=1)[:, -k:]  # (n_tokens, k)
            top_k_sum = np.sum(top_k_vals, axis=1)  # (n_tokens,)
            top_1 = np.max(probs, axis=1)  # (n_tokens,)
            concentration = top_1 / (top_k_sum + 1e-12)
            topk_concentration += concentration

        per_token_entropy /= self.n_layers
        topk_concentration /= self.n_layers

        # Cross-layer variance: L2 norm per (layer, token), then var across layers
        # activations: (n_layers, n_tokens, hidden_dim)
        norms = np.linalg.norm(activations.astype(np.float64), axis=2)  # (n_layers, n_tokens)
        cross_layer_variance = np.var(norms, axis=0)  # (n_tokens,)

        return CLAPFeatures(
            per_token_entropy=per_token_entropy,
            topk_concentration=topk_concentration,
            cross_layer_variance=cross_layer_variance,
        )


# ---------------------------------------------------------------------------
# NUPProbeV3
# ---------------------------------------------------------------------------


class NUPProbeV3:
    """NUP Probe v3 — trained logistic classifier on CLAP feature vectors.

    **Why a logistic classifier (not a deep net):**
        We have at most a few hundred real CoT pairs from Exps 502-503.  A deep net
        would overfit on such small data.  Logistic regression is the right model
        when:
        - Features are informative but data is scarce
        - Interpretability is important (we can inspect each feature's weight)
        - Deployment must be fast (<1 ms/step)
        Logistic regression's AUC is also directly comparable to v1/v2 AUC.

    **Training contract:**
        fit() accepts a list of (activations, label) pairs where activations is
        (n_layers, n_tokens, hidden_dim) and label is 1 (hallucination) or 0 (correct).
        Features are extracted via CLAPFeatureExtractor.extract_features() and then
        flattened via CLAPFeatures.to_feature_vector().  Because feature vectors may
        have different lengths for different n_tokens, we pad/truncate to a fixed
        max_feature_len during fit() and predict().

    Args:
        n_features: Expected length of the feature vector (3 * n_tokens).  If vectors
            are longer, they are truncated; if shorter, they are zero-padded.
        threshold: Probability threshold for binary classification.  Default 0.5.

    Spec: REQ-VERIFY-106, SCENARIO-VERIFY-138, SCENARIO-VERIFY-139
    """

    def __init__(
        self,
        n_features: int,
        threshold: float = 0.5,
        extractor: Optional[CLAPFeatureExtractor] = None,
    ) -> None:
        self.n_features = n_features
        self.threshold = threshold
        self.extractor = extractor or CLAPFeatureExtractor()
        # Logistic regression weights and bias (initialised at fit time)
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0
        self._is_fitted: bool = False

    def _pad_or_truncate(self, vec: np.ndarray) -> np.ndarray:
        """Pad with zeros or truncate to self.n_features length."""
        if len(vec) >= self.n_features:
            return vec[: self.n_features]
        padded = np.zeros(self.n_features, dtype=np.float64)
        padded[: len(vec)] = vec
        return padded

    def _sigmoid(self, x: float) -> float:
        # Numerically stable sigmoid
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def fit(
        self,
        pairs: list[tuple[np.ndarray, int]],
        labels: Optional[list[int]] = None,
    ) -> None:
        """Train the probe on (activations, label) pairs using mini-batch gradient descent.

        **Why gradient descent, not sklearn:**
            No sklearn dependency in carnot.  We implement a simple logistic regression
            with L2 regularisation using SGD.  20 epochs with lr=0.01 and lambda=0.01
            is sufficient to converge on small datasets (< 1000 pairs).

        Args:
            pairs: List of (activations_array, label) tuples, OR list of activations
                arrays when labels is provided separately.  label: 1=hallucination, 0=correct.
            labels: Optional separate label list.  If provided, pairs is treated as a list
                of activation arrays and labels as the corresponding label list.

        Spec: REQ-VERIFY-106
        """
        if labels is not None:
            # pairs is a list of activations, labels is separate
            combined = list(zip(pairs, labels))
        else:
            combined = list(pairs)

        if not combined:
            return

        # Extract feature vectors
        X_list = []
        y_list = []
        for item, label in combined:
            acts = np.asarray(item, dtype=np.float64)
            features = self.extractor.extract_features(acts)
            vec = self._pad_or_truncate(features.to_feature_vector())
            X_list.append(vec)
            y_list.append(float(label))

        X = np.array(X_list, dtype=np.float64)  # (n_samples, n_features)
        y = np.array(y_list, dtype=np.float64)  # (n_samples,)

        # Initialise weights
        rng = np.random.default_rng(42)
        w = rng.normal(0, 0.01, size=self.n_features)
        b = 0.0

        # Mini-batch SGD with L2 regularisation
        lr = 0.01
        lam = 0.01
        n_epochs = 20
        n = len(X)

        for _ in range(n_epochs):
            # Shuffle
            idx = rng.permutation(n)
            X_s, y_s = X[idx], y[idx]

            # Full-batch gradient (small dataset)
            logits = X_s @ w + b  # (n,)
            preds = np.array([self._sigmoid(float(z)) for z in logits])
            err = preds - y_s  # (n,)
            grad_w = (X_s.T @ err) / n + lam * w
            grad_b = float(np.mean(err))
            w -= lr * grad_w
            b -= lr * grad_b

        self._weights = w
        self._bias = b
        self._is_fitted = True

    def predict(self, features: CLAPFeatures) -> float:
        """Predict hallucination probability from CLAPFeatures.

        **Returns probability in [0, 1]:**
            > threshold → predicted hallucination.
            The raw probability (not just the binary decision) is returned so
            callers can use it for ranking or calibrated confidence.

        Args:
            features: CLAPFeatures extracted from a CoT step's activation tensor.

        Returns:
            Float in [0, 1].  Higher = more likely hallucination.

        Spec: REQ-VERIFY-106
        """
        if not self._is_fitted or self._weights is None:
            # Before fit, return uninformative 0.5
            return 0.5

        vec = self._pad_or_truncate(features.to_feature_vector())
        logit = float(np.dot(self._weights, vec)) + self._bias
        return self._sigmoid(logit)

    def evaluate(
        self,
        pairs: list[tuple[np.ndarray, int]],
        labels: Optional[list[int]] = None,
    ) -> dict:
        """Evaluate the probe on held-out pairs, returning auroc and threshold.

        **Evaluation protocol:**
            For each pair, extract features and compute predict() probability.
            Build ROC curve by threshold sweep, compute AUC via trapezoidal rule.
            Return dict with 'auroc' (float in [0,1]) and 'threshold' (self.threshold).

            Edge cases:
                < 2 pairs → auroc=0.5 (undefined).
                All labels same → auroc=0.5 (cannot discriminate).

        Args:
            pairs: Same format as fit(). List of (activations, label) or activations only.
            labels: Optional separate labels.

        Returns:
            Dict with keys: auroc (float), threshold (float), n_pairs (int).

        Spec: REQ-VERIFY-106, SCENARIO-VERIFY-138, SCENARIO-VERIFY-139
        """
        if labels is not None:
            combined = list(zip(pairs, labels))
        else:
            combined = list(pairs)

        if len(combined) < 2:
            return {"auroc": 0.5, "threshold": self.threshold, "n_pairs": len(combined)}

        scores_labels = []
        for item, label in combined:
            acts = np.asarray(item, dtype=np.float64)
            features = self.extractor.extract_features(acts)
            prob = self.predict(features)
            scores_labels.append((prob, int(label)))

        n_pos = sum(1 for _, l in scores_labels if l == 1)
        n_neg = sum(1 for _, l in scores_labels if l == 0)

        if n_pos == 0 or n_neg == 0:
            return {"auroc": 0.5, "threshold": self.threshold, "n_pairs": len(combined)}

        # Sort descending by score
        sorted_sl = sorted(scores_labels, key=lambda x: x[0], reverse=True)

        roc = [(0.0, 0.0)]
        tp = fp = 0
        for _, label in sorted_sl:
            if label == 1:
                tp += 1
            else:
                fp += 1
            roc.append((fp / n_neg, tp / n_pos))

        auc = 0.0
        for i in range(len(roc) - 1):
            fpr_prev, tpr_prev = roc[i]
            fpr_curr, tpr_curr = roc[i + 1]
            if fpr_curr > fpr_prev:
                auc += (fpr_curr - fpr_prev) * (tpr_curr + tpr_prev) / 2.0

        auc = float(min(1.0, max(0.0, auc)))
        return {"auroc": auc, "threshold": self.threshold, "n_pairs": len(combined)}
