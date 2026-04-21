"""HALPProbe — Pre-generative hallucination detection via question-end hidden-state probing.

**Why HALPProbe exists (arXiv 2603.05465, HALP):**
    Standard hallucination detection runs AFTER the model has generated its answer.
    This is expensive: you must first generate tokens, then score them.

    HALP's key insight (2603.05465): at the moment the model finishes reading the
    question — before it generates the first output token — its hidden state already
    encodes whether it will hallucinate.  A small MLP trained on those "question-end"
    hidden states achieves 0.93 AUROC on hallucination detection without producing
    any output at all.

    This makes HALP a Tier 0g probe: it runs BEFORE every other tier in the cascade.
    If the probe predicts hallucination at query time, the system can early-exit to
    repair before generation even begins, saving both latency and compute.

**CPU-only proxy implementation:**
    A real HALP probe requires access to the LLM's internal hidden states.
    On CPU-only hardware, we substitute word-length features from the question text
    as a proxy for the question-end hidden state.  This is acknowledged as an
    approximation — the point of Exp 663 is to validate the pipeline architecture
    and measure whether even basic question features carry any hallucination signal.

**Architecture:**
    - Feature extractor: word-length vector of dimension n_features
    - Probe: linear layer (weights + bias) trained with logistic regression via Adam
    - Threshold: configurable, default 0.5

Spec: REQ-VERIFY-155, SCENARIO-VERIFY-209, SCENARIO-VERIFY-210
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# HALPProbeResult
# ---------------------------------------------------------------------------


@dataclass
class HALPProbeResult:
    """Result from a single HALPProbe prediction call.

    **Fields:**
        question: The input question text that was probed.
        hidden_state_dim: Dimension of the feature vector used as the hidden-state proxy.
        hallucination_score: Float in [0, 1] — probability that the model will hallucinate.
            Higher = more likely to hallucinate.
        predicted_hallucinated: True when hallucination_score >= probe threshold.

    Spec: REQ-VERIFY-155
    """

    question: str
    hidden_state_dim: int
    hallucination_score: float
    predicted_hallucinated: bool


# ---------------------------------------------------------------------------
# HALPProbe
# ---------------------------------------------------------------------------


class HALPProbe:
    """Pre-generative hallucination probe based on HALP (arXiv 2603.05465).

    **How it works:**
        1. At query time, extract a feature vector from the question text.
           On real hardware this would be the transformer's last hidden state after
           reading the question; on CPU we use normalised word lengths as a proxy.
        2. Apply a linear probe (weights + bias) to produce a logit.
        3. Sigmoid the logit to get a hallucination probability.
        4. Return HALPProbeResult with predicted_hallucinated = (score >= threshold).

    **Why a linear probe:**
        The original HALP paper uses a small MLP.  For CPU-only evaluation, a linear
        probe is sufficient to test whether the feature space carries any signal.
        If it does, the architecture is sound and a deeper probe on real hidden states
        will perform even better.

    Args:
        n_features: Dimension of the feature vector (proxy hidden-state size). Default 64.
        hidden_dim: Unused in this linear implementation; kept for API compatibility
            with the MLP described in the paper. Default 32.
        threshold: Hallucination score above which predicted_hallucinated is True. Default 0.5.

    Spec: REQ-VERIFY-155, SCENARIO-VERIFY-209, SCENARIO-VERIFY-210
    """

    def __init__(
        self,
        n_features: int = 64,
        hidden_dim: int = 32,
        threshold: float = 0.5,
    ) -> None:
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.weights: dict[str, Any] | None = None

    def _extract_features(self, question: str) -> Any:
        """Extract question-end hidden-state proxy features (no actual LLM needed for CPU).

        **Proxy strategy:**
            We use per-word character length, normalised to [0, 1] by dividing by 20.
            This captures surface-level complexity of the question's vocabulary without
            requiring a tokeniser or GPU.  The last n_features words are packed into
            the tail of the feature vector so that the question-end context dominates —
            mirroring how the real HALP probe focuses on the hidden state AFTER the
            last question token.

        Args:
            question: Input question text.

        Returns:
            JAX array of shape (n_features,) with values in [0, 1].

        Spec: REQ-VERIFY-155
        """
        import jax.numpy as jnp

        words = question.split()
        n_words = min(len(words), self.n_features)
        features = jnp.zeros(self.n_features)
        for i, word in enumerate(words[-n_words:]):
            idx = self.n_features - n_words + i
            features = features.at[idx].set(float(len(word)) / 20.0)
        return features

    def train(self, questions: list[str], labels: list[int]) -> dict[str, Any]:
        """Train a linear probe on question features using Adam optimiser.

        **Training procedure:**
            1. Extract feature vectors for all training questions.
            2. Initialise probe weights and bias to zero.
            3. Run 100 Adam steps minimising sigmoid binary cross-entropy loss.
            4. Store trained weights in self.weights for use by predict().

        **Why Adam over SGD:**
            Adam's adaptive learning rate handles the sparse feature matrix well —
            most dimensions are zero for short questions, which causes SGD to oscillate.
            Adam accumulates second moments per-parameter and scales updates accordingly.

        Args:
            questions: List of question strings.
            labels: Binary labels (1 = hallucinated, 0 = correct), same length as questions.

        Returns:
            Dict with 'weights' (list of floats) and 'bias' (list with one float).

        Spec: REQ-VERIFY-155, SCENARIO-VERIFY-209
        """
        import jax
        import jax.numpy as jnp
        import optax

        feature_matrix = jnp.stack([self._extract_features(q) for q in questions])
        labels_array = jnp.array(labels, dtype=jnp.float32)

        weights = jnp.zeros(self.n_features)
        bias = jnp.zeros(1)

        # Initialise optimizer state for BOTH parameters together so that the
        # pytree structure matches the gradient tuple returned by jax.grad.
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init((weights, bias))

        def loss_fn(
            w: Any, b: Any, X: Any, y: Any
        ) -> Any:
            logits = X @ w + b[0]
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))

        for _ in range(100):
            grads = jax.grad(loss_fn, argnums=(0, 1))(
                weights, bias, feature_matrix, labels_array
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            weights, bias = optax.apply_updates((weights, bias), updates)

        self.weights = {"weights": weights.tolist(), "bias": bias.tolist()}
        return self.weights

    def predict(self, question: str) -> HALPProbeResult:
        """Predict hallucination probability from question-end hidden state.

        **Prediction logic:**
            If the probe has not been trained (self.weights is None), fall back to
            the mean of the feature vector as a naive score.  This ensures the probe
            is safe to call before training (e.g., during zero-shot evaluation).

            After training, apply the linear probe: sigmoid(features @ weights + bias).

        Args:
            question: Input question text.

        Returns:
            HALPProbeResult with hallucination_score in [0, 1] and predicted_hallucinated
            set to True when the score meets or exceeds self.threshold.

        Spec: REQ-VERIFY-155, SCENARIO-VERIFY-210
        """
        import jax
        import jax.numpy as jnp

        features = self._extract_features(question)

        if self.weights is None:
            # Untrained probe: use feature mean as a naive, uninformative score.
            score = float(jnp.mean(features))
        else:
            w = jnp.array(self.weights["weights"])
            b = jnp.array(self.weights["bias"])
            logit = features @ w + b[0]
            score = float(jax.nn.sigmoid(logit))

        return HALPProbeResult(
            question=question,
            hidden_state_dim=self.n_features,
            hallucination_score=score,
            predicted_hallucinated=(score >= self.threshold),
        )
