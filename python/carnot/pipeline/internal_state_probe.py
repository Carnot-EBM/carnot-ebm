"""InternalStateProbe — lightweight linear probe on LLM hidden states for Tier 2 credibility.

**Researcher summary (arXiv 2511.06209):**
    A single linear layer trained on (hidden_state, is_correct_label) pairs matches
    PRMs 810x larger for step-level reasoning credibility.  Carnot's EORM is 55M params.
    If the probe achieves comparable AUC on FOVER pairs, it should replace EORM as the
    default Tier 2 because:

    - **Zero extra inference cost**: hidden states are produced by the LLM during the
      forward pass anyway — we just attach a 1-parameter-per-hidden-dim linear layer.
    - **810x smaller**: EORM is ~55M params; a probe on a 1024-dim hidden state is
      1024 + 1 = 1025 params.
    - **Simpler training**: standard BCE on (hidden_state, is_correct_label) pairs, no
      contrastive margin needed.

**Design (arXiv 2511.06209 §3):**
    The probe reads the hidden state at a fixed layer index (default -4, i.e. 4th from
    last).  Empirically, intermediate layers carry more verifiable-reasoning signal than
    the final layer — the final layer collapses to next-token prediction and loses the
    "am I on track" signal that intermediate layers preserve.

**Data flow:**
    1. LLM produces (hidden_state_at_layer_N) for a reasoning step.
    2. Probe: logit = W @ hidden_state + b.
    3. Sigmoid to [0, 1] — higher = more likely INCORRECT.
    4. In Tier 2 of ThreeTierPipeline: if probe.score(hidden_state) > threshold,
       flag the step for EORM re-ranking or Ising repair.

Spec: REQ-VERIFY-115, SCENARIO-VERIFY-151, SCENARIO-VERIFY-152, SCENARIO-VERIFY-153
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# InternalStateProbeResult
# ---------------------------------------------------------------------------


@dataclass
class InternalStateProbeResult:
    """Summary statistics from InternalStateProbe vs EORM comparison.

    **For engineers:**
        All fields map directly to the artifact schema so the conductor can
        parse them without special casing.  ``is_tier2_viable`` is the
        decision bit: True iff probe_auc >= 0.700 (same threshold as NUPProbeV4
        in REQ-VERIFY-110).  ``honest_verdict`` is a human-readable string that
        explains WHY the decision was made.

    Spec: REQ-VERIFY-115
    """

    probe_layer: int
    """Which LLM layer the hidden state was taken from (negative = from end)."""

    n_train_pairs: int
    """Number of (hidden_state, label) pairs used for probe training."""

    n_test_pairs: int
    """Number of (hidden_state, label) pairs held out for evaluation."""

    probe_auc: float
    """Area under the ROC curve for the linear probe on the test split."""

    eorm_auc: float
    """Area under the ROC curve for EORM on the same test split (baseline)."""

    param_count_ratio: float
    """probe_params / eorm_params — expected ~1/810 = 0.00123 per arXiv 2511.06209."""

    is_tier2_viable: bool
    """True iff probe_auc >= 0.700 (REQ-VERIFY-110 threshold reused for consistency)."""

    honest_verdict: str
    """One of: 'probe_tier2_viable' | 'probe_below_threshold' | 'synthetic_proxy'."""


# ---------------------------------------------------------------------------
# InternalStateProbe
# ---------------------------------------------------------------------------


class InternalStateProbe:
    """Single linear layer trained on (hidden_state, is_correct_label) pairs.

    **For engineers:**
        This is deliberately the simplest possible model — a logistic regression
        over LLM hidden states.  The design is justified by arXiv 2511.06209,
        which shows that the verifiable-reasoning signal is linearly separable in
        intermediate hidden states.  A non-linear probe (MLP) does not improve AUC
        meaningfully in their ablation (Table 3), so we keep it simple.

        Architecture::

            logit = W @ hidden_state + b          # shape: (1,)
            p_incorrect = sigmoid(logit)           # higher = more likely wrong

        Training uses BCE loss with mini-batch gradient descent.  No regularisation
        is needed at this scale because the probe is so small it cannot overfit to
        the ~50–200 FOVER pairs we have.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the LLM hidden state (e.g. 1024 for a 7B model).
    probe_layer : int
        Which layer to probe (negative = from end, e.g. -4 = 4th from last).

    Spec: REQ-VERIFY-115, SCENARIO-VERIFY-151
    """

    def __init__(self, hidden_size: int = 1024, probe_layer: int = -4) -> None:
        self.hidden_size = hidden_size
        self.probe_layer = probe_layer

        # Xavier-uniform initialisation: keeps gradients in reasonable range
        # at the start of training regardless of hidden_size.
        scale = math.sqrt(6.0 / (hidden_size + 1))
        rng = np.random.default_rng(42)
        self._W: np.ndarray = rng.uniform(-scale, scale, size=(hidden_size,)).astype(np.float64)
        self._b: float = 0.0

    # ------------------------------------------------------------------
    # train()
    # ------------------------------------------------------------------

    def train(
        self,
        pairs: list[tuple[np.ndarray, int]],
        *,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        """Fit the probe on (hidden_state, is_incorrect_label) pairs via BCE.

        **For engineers:**
            We use full-batch gradient descent (not mini-batch) because our
            datasets are small (~50–200 pairs).  The BCE gradient with respect
            to W is::

                dL/dW = (1/N) * sum_i { (p_i - y_i) * x_i }
                dL/db = (1/N) * sum_i { p_i - y_i }

            where p_i = sigmoid(W @ x_i + b) and y_i ∈ {0, 1}.
            Label convention: 1 = INCORRECT (the probe should output a HIGH
            logit for steps that are wrong, so the caller can flag them).

        Parameters
        ----------
        pairs : list of (hidden_state, label)
            ``label`` is 1 if the step is INCORRECT, 0 if CORRECT.
        epochs : int
            Number of full passes over the training data.
        lr : float
            Learning rate for gradient descent.

        Spec: SCENARIO-VERIFY-151
        """
        if not pairs:
            return

        X = np.stack([p[0] for p in pairs], axis=0).astype(np.float64)  # (N, D)
        y = np.array([p[1] for p in pairs], dtype=np.float64)             # (N,)

        for _ in range(epochs):
            logits = X @ self._W + self._b          # (N,)
            # Numerically stable sigmoid
            probs = _sigmoid(logits)
            error = probs - y                        # (N,)
            self._W -= lr * (X.T @ error) / len(y)
            self._b -= lr * error.mean()

    # ------------------------------------------------------------------
    # score()
    # ------------------------------------------------------------------

    def score(self, hidden_state: np.ndarray) -> float:
        """Return the probability that this step is INCORRECT.

        Higher = more likely incorrect = stronger Tier 2 flag.
        The raw logit is passed through sigmoid so the output is always in [0, 1].

        Parameters
        ----------
        hidden_state : np.ndarray
            1-D array of shape (hidden_size,).

        Returns
        -------
        float in [0, 1]

        Spec: SCENARIO-VERIFY-152
        """
        hs = np.asarray(hidden_state, dtype=np.float64).ravel()
        logit = float(hs @ self._W + self._b)
        return float(_sigmoid(np.array(logit)))

    # ------------------------------------------------------------------
    # param_count
    # ------------------------------------------------------------------

    @property
    def param_count(self) -> int:
        """Total trainable parameters: hidden_size weights + 1 bias."""
        return self.hidden_size + 1


# ---------------------------------------------------------------------------
# simulate_hidden_states()
# ---------------------------------------------------------------------------


def simulate_hidden_states(
    n_samples: int,
    hidden_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic hidden states for CI testing without a real LLM.

    **For engineers:**
        Real LLM hidden states cluster by correctness in intermediate layers
        (the entire premise of arXiv 2511.06209).  We simulate this by:

        - Correct steps → low-norm Gaussian (mean norm ≈ 1.0).
        - Incorrect steps → high-norm Gaussian (mean norm ≈ 2.5).

        This is enough signal for a linear probe to learn non-trivially, so
        the CI test validates that the probe *can* train and score — not that
        it achieves a specific AUC on synthetic data.

    Parameters
    ----------
    n_samples : int
        Number of samples per class (total = 2 * n_samples).
    hidden_size : int
        Dimensionality of each hidden state.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    correct_states : np.ndarray of shape (n_samples, hidden_size)
    incorrect_states : np.ndarray of shape (n_samples, hidden_size)

    Spec: SCENARIO-VERIFY-152
    """
    rng = np.random.default_rng(seed)

    # Correct steps: unit-norm neighbourhood — tight cluster around the origin
    correct_raw = rng.normal(0.0, 1.0, size=(n_samples, hidden_size))
    # Normalise to norm ≈ 1.0 then add small noise
    norms = np.linalg.norm(correct_raw, axis=1, keepdims=True) + 1e-9
    correct_states = correct_raw / norms + rng.normal(0.0, 0.1, size=(n_samples, hidden_size))

    # Incorrect steps: higher norm, larger spread — further from correct cluster
    incorrect_raw = rng.normal(0.0, 1.5, size=(n_samples, hidden_size))
    norms2 = np.linalg.norm(incorrect_raw, axis=1, keepdims=True) + 1e-9
    incorrect_states = (
        2.5 * incorrect_raw / norms2 + rng.normal(0.0, 0.3, size=(n_samples, hidden_size))
    )

    return correct_states.astype(np.float64), incorrect_states.astype(np.float64)


# ---------------------------------------------------------------------------
# evaluate_probe_vs_eorm()
# ---------------------------------------------------------------------------


def evaluate_probe_vs_eorm(
    probe: InternalStateProbe,
    eorm_scores: list[float],
    test_pairs: list[tuple[np.ndarray, int]],
    eorm_param_count: int = 55_000_000,
) -> InternalStateProbeResult:
    """Compute AUC for probe and EORM, return a structured comparison result.

    **For engineers:**
        AUC is computed via the trapezoid rule on the empirical ROC curve.
        We sort by score descending and compute TPR/FPR at each threshold.
        This matches sklearn.metrics.roc_auc_score but avoids the dependency.

        ``eorm_scores`` are pre-computed scores for the same test_pairs, in
        the same order, with the same convention: higher = more likely incorrect.

        ``param_count_ratio`` = probe.param_count / eorm_param_count.
        For a 1024-dim probe vs 55M EORM: 1025 / 55_000_000 ≈ 0.0000186.
        arXiv 2511.06209 uses "810x smaller" as their headline number because
        they compare against 810M-param PRMs; we preserve 0.00123 as the
        paper's ratio but compute our own actual ratio too.

    Parameters
    ----------
    probe : InternalStateProbe
        Trained probe.
    eorm_scores : list[float]
        Pre-computed EORM scores for each test pair (same order, higher = incorrect).
    test_pairs : list of (hidden_state, label)
        Test set; label=1 means INCORRECT.
    eorm_param_count : int
        EORM parameter count for ratio calculation (default 55M).

    Returns
    -------
    InternalStateProbeResult

    Spec: SCENARIO-VERIFY-153
    """
    if not test_pairs:
        return InternalStateProbeResult(
            probe_layer=probe.probe_layer,
            n_train_pairs=0,
            n_test_pairs=0,
            probe_auc=0.5,
            eorm_auc=0.5,
            param_count_ratio=probe.param_count / eorm_param_count,
            is_tier2_viable=False,
            honest_verdict="synthetic_proxy",
        )

    probe_scores = [probe.score(hs) for hs, _ in test_pairs]
    labels = [label for _, label in test_pairs]

    probe_auc = _compute_auc(probe_scores, labels)
    eorm_auc = _compute_auc(eorm_scores, labels)
    param_ratio = probe.param_count / eorm_param_count

    # Honest verdict logic:
    # - 'synthetic_proxy': when all labels are the same class (AUC is undefined / 0.5)
    # - 'probe_tier2_viable': probe_auc >= 0.700
    # - 'probe_below_threshold': probe_auc < 0.700
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        honest_verdict = "synthetic_proxy"
        is_viable = False
    elif probe_auc >= 0.700:
        honest_verdict = "probe_tier2_viable"
        is_viable = True
    else:
        honest_verdict = "probe_below_threshold"
        is_viable = False

    return InternalStateProbeResult(
        probe_layer=probe.probe_layer,
        n_train_pairs=0,  # caller fills n_train_pairs via experiment script
        n_test_pairs=len(test_pairs),
        probe_auc=round(probe_auc, 4),
        eorm_auc=round(eorm_auc, 4),
        param_count_ratio=round(param_ratio, 8),
        is_tier2_viable=is_viable,
        honest_verdict=honest_verdict,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: avoids overflow for large positive x."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _compute_auc(scores: list[float], labels: list[int]) -> float:
    """Compute AUC via trapezoidal rule on the empirical ROC curve.

    **For engineers:**
        Standard algorithm: sort by score descending (treat high score = positive),
        accumulate TP/FP counts, compute TPR/FPR at each threshold point, then
        integrate.  This is equivalent to sklearn.metrics.roc_auc_score but has
        no external dependency.

        Returns 0.5 (random baseline) when all labels are the same class.
    """
    n = len(labels)
    if n == 0:
        return 0.5

    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by score descending
    pairs_sorted = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    tp, fp = 0, 0
    auc = 0.0
    prev_fpr = 0.0

    for _, label in pairs_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        # Trapezoid step: area = tpr * (fpr - prev_fpr)
        auc += tpr * (fpr - prev_fpr)
        prev_fpr = fpr

    return float(auc)
