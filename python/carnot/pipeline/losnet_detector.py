"""LOS-Net: sequence-level hallucination detector over full output token distributions.

**Researcher summary:**
    arXiv 2503.14043 (LOS-Net) shows that the FULL trajectory of next-token probability
    distributions carries far more hallucination signal than any single token's entropy.
    SpilledEnergyDetector (Tier 0b) scores one token at a time. LOS-Net instead builds
    a feature vector over the SEQUENCE of per-step entropy values, capturing patterns
    like low→high→low entropy ("garden-path" hallucinations where the model commits to
    a wrong path and doubles down) that are invisible to per-token measures.

**Why sequence-level entropy trajectory?**
    When a model hallucinates a fact it is not confident about, entropy typically spikes
    mid-generation at the point of commitment (high uncertainty) and then collapses again
    once the model has invented its answer (false confidence). A per-token view sees only
    "some high-entropy tokens"; a sequence-level view captures the PATTERN — the rise and
    fall — which is the true hallucination fingerprint.

**Three features used by LOSNetClassifier:**
    1. entropy_variance: high variance means the model oscillated between confident and
       uncertain states — a sign of a garden-path trajectory.
    2. entropy_trend: a positive slope means entropy INCREASED over the generation, a sign
       the model became progressively more uncertain (hallucination risk grows toward end).
       A negative slope means the model started uncertain and got confident (normal recall).
    3. max_entropy: absolute worst-case uncertainty at any point in the sequence.

**Why < 5M params?**
    We need this to run as a Tier 0h pre-filter BEFORE the expensive KB lookups (Tier 1+).
    A 3-feature linear classifier has ~4 parameters — orders of magnitude under the cap.
    Keeping it tiny also makes the features the important thing, not over-fitting.

**Relationship to SpilledEnergyDetector (Tier 0b):**
    SpilledEnergyDetector uses the log-sum-exp minus expected-logit formula to get a
    per-token scalar. LOSNetClassifier wraps those scalars into sequence-level statistics.
    The two detectors are complementary: Tier 0b catches globally high-spill sequences;
    Tier 0h (this module) catches locally-patterned trajectories that Tier 0b misses.

Spec: REQ-VERIFY-153, REQ-VERIFY-154,
      SCENARIO-VERIFY-202, SCENARIO-VERIFY-203, SCENARIO-VERIFY-204
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# LOSNetFeatures — the sequence-level feature vector
# ---------------------------------------------------------------------------


@dataclass
class LOSNetFeatures:
    """Sequence-level hallucination features derived from a token distribution trajectory.

    **Researcher summary:**
        Captures the shape of the entropy sequence H_0, H_1, ..., H_{T-1} produced
        by one LLM generation. These three scalars summarise the trajectory in a form
        that a small linear classifier can score.

    **Detailed explanation for engineers:**
        top_k_probs: the raw per-step top-K probability vectors (kept for diagnostics).
        sequence_entropy: H_t = -sum(p * log(p)) at each step (in nats).
        entropy_variance: var(H) across the sequence — measures oscillation.
        entropy_trend: least-squares slope of H over time — positive means entropy
            grew (model got less confident), negative means it shrank (model got
            more confident as it generated more tokens).

    Attributes:
        top_k_probs: per-step top-K probabilities; shape (n_steps, top_k).
        n_steps: number of token positions in the generation.
        sequence_entropy: per-step Shannon entropy H_t (nats).
        entropy_variance: variance of sequence_entropy.
        entropy_trend: linear slope of sequence_entropy vs. position index.

    Spec: REQ-VERIFY-153
    """

    top_k_probs: list[list[float]]
    n_steps: int
    sequence_entropy: list[float]
    entropy_variance: float
    entropy_trend: float


# ---------------------------------------------------------------------------
# extract_losnet_features — compute LOSNetFeatures from a logit sequence
# ---------------------------------------------------------------------------


def extract_losnet_features(
    logit_sequences: list[list[float]],
    top_k: int = 10,
) -> LOSNetFeatures:
    """Extract LOS-Net features from a sequence of per-step probability distributions.

    **Researcher summary:**
        Converts a raw token distribution trajectory into the three scalar features
        (entropy_variance, entropy_trend, max_entropy) used by LOSNetClassifier.
        Handles both full vocab distributions and pre-filtered top-K inputs.

    **Detailed explanation for engineers:**
        logit_sequences[t] can be either:
        - A full vocab probability vector (length V, sums to ~1.0), OR
        - A top-K probability vector (length <= top_k, sums to <= 1.0).

        In either case we renormalise within each step so that entropy is computed
        over a valid probability distribution. We also clamp probabilities to [1e-10, 1]
        to avoid log(0).

        The function does NOT require JAX — it is pure Python so it can run in CI
        without GPU hardware.

        entropy_trend is computed as the slope of the ordinary least-squares regression
        of H_t on t. A positive slope means entropy grew over the generation.

    Args:
        logit_sequences: list of length n_steps; each inner list is a probability vector
            (does not need to sum to 1 — renormalised internally).
        top_k: how many entries to retain per step when the input has more than top_k
            values. Values are sorted descending and the top_k are kept.

    Returns:
        LOSNetFeatures with all fields populated.

    Spec: REQ-VERIFY-153, SCENARIO-VERIFY-202
    """
    n_steps = len(logit_sequences)
    if n_steps == 0:
        return LOSNetFeatures(
            top_k_probs=[],
            n_steps=0,
            sequence_entropy=[],
            entropy_variance=0.0,
            entropy_trend=0.0,
        )

    top_k_probs: list[list[float]] = []
    sequence_entropy: list[float] = []

    for step_probs in logit_sequences:
        # Keep top_k values if step has more than top_k entries.
        if len(step_probs) > top_k:
            sorted_probs = sorted(step_probs, reverse=True)[:top_k]
        else:
            sorted_probs = list(step_probs)

        # Renormalise so the slice sums to 1.0.
        total = sum(sorted_probs)
        if total <= 0.0:
            # Degenerate case: all zeros — treat as uniform over the slice.
            k = max(len(sorted_probs), 1)
            sorted_probs = [1.0 / k] * k
            total = 1.0
        normalised = [p / total for p in sorted_probs]
        top_k_probs.append(normalised)

        # Shannon entropy H_t = -sum(p * log(p)) in nats.
        h_t = -sum(p * math.log(max(p, 1e-10)) for p in normalised)
        sequence_entropy.append(h_t)

    # Variance of the entropy trajectory.
    mean_h = sum(sequence_entropy) / n_steps
    entropy_variance = sum((h - mean_h) ** 2 for h in sequence_entropy) / n_steps

    # Linear trend (OLS slope of H_t vs. t).
    # slope = (n * sum(t*H_t) - sum(t)*sum(H_t)) / (n * sum(t^2) - sum(t)^2)
    ts = list(range(n_steps))
    sum_t = sum(ts)
    sum_h = sum(sequence_entropy)
    sum_th = sum(t * h for t, h in zip(ts, sequence_entropy))
    sum_t2 = sum(t * t for t in ts)
    denom = n_steps * sum_t2 - sum_t * sum_t
    if denom == 0.0 or n_steps == 1:
        entropy_trend = 0.0
    else:
        entropy_trend = (n_steps * sum_th - sum_t * sum_h) / denom

    return LOSNetFeatures(
        top_k_probs=top_k_probs,
        n_steps=n_steps,
        sequence_entropy=sequence_entropy,
        entropy_variance=entropy_variance,
        entropy_trend=entropy_trend,
    )


# ---------------------------------------------------------------------------
# LOSNetClassifier — small linear classifier over the three features
# ---------------------------------------------------------------------------


class LOSNetClassifier:
    """Lightweight linear classifier for sequence-level hallucination detection.

    **Researcher summary:**
        Trains on (entropy_variance, entropy_trend, max_entropy) feature vectors from
        positive (hallucinated) and negative (correct) sequences. At inference, returns
        a probability in [0, 1] that the input is a hallucination.

    **Why linear (not MLP)?**
        With only 3 features and ~50 training pairs, a deep MLP would overfit. A linear
        logistic classifier has 4 trainable parameters (3 weights + 1 bias) — it is the
        minimum-capacity model that can learn a hallucination boundary in this feature
        space. This is comfortably under the 5M-parameter budget from REQ-VERIFY-153.

    **Training algorithm:**
        Gradient descent on binary cross-entropy loss (logistic regression) with 500
        iterations and learning rate 0.01. No JAX required — pure Python so CI runs
        without GPU hardware.

    **Prediction:**
        score(features) returns sigmoid(w^T x + b) where x = [variance, trend, max_entropy].

    Spec: REQ-VERIFY-153, REQ-VERIFY-154,
          SCENARIO-VERIFY-203, SCENARIO-VERIFY-204
    """

    def __init__(self, n_features: int = 3) -> None:
        """Create an untrained LOSNetClassifier.

        Args:
            n_features: number of input features (default 3: variance, trend, max_entropy).
                The classifier currently always uses exactly 3 features regardless of
                this parameter, but the parameter is retained for API symmetry with
                the LOS-Net paper's variable-feature formulation.
        """
        # Weight vector and bias, initialised to zero (untrained state).
        self._weights: list[float] = [0.0] * n_features
        self._bias: float = 0.0
        self._trained: bool = False
        self._n_features = n_features

    # ------------------------------------------------------------------
    # _extract_vector — convert LOSNetFeatures to a plain Python list
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_vector(features: LOSNetFeatures) -> list[float]:
        """Return the fixed-size feature vector [variance, trend, max_entropy].

        Why max_entropy rather than mean_entropy: max is more sensitive to the
        brief entropy spikes that characterise garden-path hallucinations (the
        model "almost" commits to the wrong token at one position). Mean would
        dilute this spike signal.
        """
        max_entropy = max(features.sequence_entropy) if features.sequence_entropy else 0.0
        return [features.entropy_variance, features.entropy_trend, max_entropy]

    # ------------------------------------------------------------------
    # _sigmoid
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Numerically stable sigmoid σ(z) = 1 / (1 + exp(-z))."""
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        # Avoid overflow for large negative z.
        e = math.exp(z)
        return e / (1.0 + e)

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------

    def train(
        self,
        positive_features: list[LOSNetFeatures],
        negative_features: list[LOSNetFeatures],
        *,
        learning_rate: float = 0.01,
        n_iterations: int = 500,
    ) -> None:
        """Fit the linear classifier via gradient descent on binary cross-entropy.

        **Detailed explanation for engineers:**
            Positive examples (hallucinations) have label = 1.
            Negative examples (correct outputs) have label = 0.
            We minimise the binary cross-entropy loss:
                L = -mean(y * log(p) + (1-y) * log(1-p))
            where p = sigmoid(w^T x + b).
            Gradient: dL/dw = mean((p - y) * x), dL/db = mean(p - y).

        Args:
            positive_features: LOSNetFeatures for hallucinated (incorrect) examples.
            negative_features: LOSNetFeatures for correct examples.
            learning_rate: step size for gradient descent.
            n_iterations: number of full-dataset gradient steps.

        Spec: REQ-VERIFY-153, SCENARIO-VERIFY-203
        """
        # Build the dataset: (feature_vector, label) pairs.
        X: list[list[float]] = []
        y: list[float] = []

        for f in positive_features:
            X.append(self._extract_vector(f))
            y.append(1.0)
        for f in negative_features:
            X.append(self._extract_vector(f))
            y.append(0.0)

        n = len(X)
        if n == 0:
            return

        nf = self._n_features
        weights = [0.0] * nf
        bias = 0.0

        for _ in range(n_iterations):
            # Forward pass: compute predictions.
            preds = [self._sigmoid(sum(weights[j] * X[i][j] for j in range(nf)) + bias) for i in range(n)]
            # Compute gradients.
            errors = [preds[i] - y[i] for i in range(n)]
            grad_w = [sum(errors[i] * X[i][j] for i in range(n)) / n for j in range(nf)]
            grad_b = sum(errors) / n
            # Update weights.
            weights = [weights[j] - learning_rate * grad_w[j] for j in range(nf)]
            bias = bias - learning_rate * grad_b

        self._weights = weights
        self._bias = bias
        self._trained = True

    # ------------------------------------------------------------------
    # score
    # ------------------------------------------------------------------

    def score(self, features: LOSNetFeatures) -> float:
        """Return P(hallucination) for one sequence's LOSNetFeatures.

        **Detailed explanation for engineers:**
            Computes sigmoid(w^T x + b) where x = [entropy_variance, entropy_trend,
            max_entropy]. Returns a value in [0, 1]: 1 means certain hallucination,
            0 means certain correct output.

            If the classifier has not been trained yet, returns 0.5 (maximum uncertainty)
            rather than raising an exception, so callers can always call score() safely.

        Args:
            features: LOSNetFeatures from extract_losnet_features().

        Returns:
            Float in [0, 1] — probability of hallucination.

        Spec: REQ-VERIFY-154, SCENARIO-VERIFY-204
        """
        if not self._trained:
            return 0.5  # untrained: maximum uncertainty
        x = self._extract_vector(features)
        z = sum(self._weights[j] * x[j] for j in range(self._n_features)) + self._bias
        return self._sigmoid(z)


# ---------------------------------------------------------------------------
# build_losnet_artifact — standardised result dict for Exp 675
# ---------------------------------------------------------------------------


def build_losnet_artifact(
    auc: float,
    vs_spilled_energy_auc: float,
    n_train_pairs: int,
    n_eval_pairs: int,
    *,
    honest_verdict: str,
    feature_importances: dict[str, float] | None = None,
) -> dict:
    """Build a JSON-serialisable artifact describing the LOS-Net evaluation result.

    **Detailed explanation for engineers:**
        The artifact records both the absolute AUC of LOSNetClassifier and its
        performance relative to SpilledEnergyDetector. This lets the conductor
        decide whether to promote LOS-Net to Tier 0h.

    Args:
        auc: AUROC of LOSNetClassifier on the held-out evaluation pairs.
        vs_spilled_energy_auc: AUROC of SpilledEnergyDetector on the same pairs.
        n_train_pairs: number of pairs used for training (80% split).
        n_eval_pairs: number of pairs used for evaluation (20% split).
        honest_verdict: "tier0h_viable" if auc >= 0.75, "below_threshold" otherwise.
        feature_importances: optional dict mapping feature name → weight magnitude.

    Returns:
        dict ready for json.dumps().

    Spec: REQ-VERIFY-153, REQ-VERIFY-154
    """
    artifact: dict = {
        "model": "LOSNetClassifier",
        "paper": "arXiv 2503.14043",
        "auc_losnet": round(auc, 4),
        "auc_spilled_energy_baseline": round(vs_spilled_energy_auc, 4),
        "auc_delta": round(auc - vs_spilled_energy_auc, 4),
        "n_train_pairs": n_train_pairs,
        "n_eval_pairs": n_eval_pairs,
        "honest_verdict": honest_verdict,
        "threshold_for_viable": 0.75,
        "features_used": ["entropy_variance", "entropy_trend", "max_entropy"],
        "n_parameters": 4,  # 3 weights + 1 bias
    }
    if feature_importances is not None:
        artifact["feature_importances"] = feature_importances
    return artifact
