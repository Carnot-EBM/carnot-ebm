"""DRIFTProbe — Tier 0i multi-layer hidden-state drift hallucination probe.

**Researcher summary:**
    arXiv 2604.13386 (Multi-Layer Probe Ensembling) shows that probing adjacent
    transformer layer pairs (L, L+1) captures a drift signal invisible to
    single-layer probes.  Hallucinating completions exhibit higher cosine distance
    between consecutive layer representations than truthful ones — the model's
    internal representations "drift" more when generating content it cannot ground
    in its training distribution.

    This module implements DRIFTProbe for Tier 0i:
        1. Accept precomputed hidden states from a model_runner callable.
        2. Compute drift signatures: per-layer-pair cosine distance (1 - cosine_sim)
           averaged over all token positions.
        3. Train a logistic regression probe on labeled (hidden_states, label) pairs
           to separate hallucinating from truthful completions.
        4. At inference time, return a calibrated violation probability.

    WHY accept hidden_states dict rather than raw text?
        Decoupling extraction from classification allows:
        a) Testing the probe logic without loading a model (inject synthetic states).
        b) Reusing extracted states across multiple probes or ablations.
        c) Swapping model_runner (real LLM, cached activations, synthetic generator)
           without changing the probe's classification logic.

    WHY logistic regression?
        The drift signal is expected to be linearly separable (per arXiv 2604.13386).
        A linear probe is interpretable — its coefficients show which layer-pair
        transition carries the most hallucination signal.  It also has very low
        overfitting risk on the 80-example training sets used in Exp 911.

    WHY default layers [-4, -3, -2, -1]?
        Last-4-layers indices are model-size-agnostic (work regardless of total depth).
        The DRIFT paper reports that late layers exhibit the largest drift gap between
        truthful and hallucinating completions, because final layers are responsible
        for "committing" to an output token — hallucination shows up as instability
        in this commitment.

Spec: REQ-TIER0-005, SCENARIO-TIER0-005
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float vectors.

    Returns 1.0 when either vector is zero (no drift, avoids division by zero).
    Padding tokens and empty activations should not inflate the drift estimate.

    Args:
        a: 1-D float array.
        b: 1-D float array, same shape as a.

    Returns:
        Cosine similarity in [-1, 1].
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# DRIFTProbe
# ---------------------------------------------------------------------------


class DRIFTProbe:
    """Tier 0i multi-layer hidden-state drift hallucination probe.

    **For engineers:**
        This probe is wired into ThreeTierPipeline.wire_drift_probe() as an
        advisory signal.  It adds an `ood_auc_drift` metric to the experiment
        artifact but does NOT short-circuit the Ising stage or change the
        verified/violated outcome.

        The probe requires a model_runner callable that accepts text (str) and
        returns a dict mapping layer index (int) to a 2-D numpy array of shape
        (seq_len, hidden_dim).  The probe then computes per-pair drift from
        those hidden states and classifies via a fitted logistic regression.

    Usage::

        runner = lambda text: my_model.get_hidden_states(text, layers=[-4,-3,-2,-1])
        probe = DRIFTProbe(model_runner=runner, layers=[-4, -3, -2, -1])
        probe.fit(correct_examples, hallucinated_examples)
        prob = probe.predict_violation_prob(hidden_states)

    Args:
        model_runner:  Callable[[str], dict[int, np.ndarray]] — runs the LLM
                       forward pass and returns hidden states keyed by layer index.
                       If None, extract_drift_signature() returns zero signatures
                       (CI-safe mode; classification still works with injected states).
        layers:        Layer indices at which to extract hidden states.
                       Default: None → resolved from n_drift_pairs as
                       [-n_drift_pairs-1, ..., -1] (last n_drift_pairs+1 layers).
        n_drift_pairs: Number of consecutive layer pairs to use.  The drift
                       signature vector has exactly this many elements.
                       Default: 3 (matches arXiv 2604.13386 configuration).

    Spec: REQ-TIER0-005
    """

    def __init__(
        self,
        model_runner: Callable[[str], dict[int, np.ndarray]] | None = None,
        layers: list[int] | None = None,
        n_drift_pairs: int = 3,
    ) -> None:
        self.model_runner = model_runner
        self.n_drift_pairs = n_drift_pairs

        # Resolve layer list: default to last (n_drift_pairs+1) layers.
        if layers is not None:
            self.layers = layers
        else:
            # e.g. n_drift_pairs=3 → layers=[-4, -3, -2, -1]
            self.layers = list(range(-(n_drift_pairs + 1), 0))

        # LogisticRegression trained by fit(); None before fit() is called.
        self._probe = None

    # ------------------------------------------------------------------
    # Core signal extraction
    # ------------------------------------------------------------------

    def extract_drift_signature(self, hidden_states: dict[int, np.ndarray]) -> np.ndarray:
        """Compute the drift signature from precomputed hidden states.

        For each consecutive pair (L, L+1) in self.layers:
            drift_i = mean_over_tokens(1.0 - cosine_sim(h_L[t], h_L+1[t]))

        The resulting vector has shape (n_drift_pairs,) and is the feature
        vector used to train and run the logistic regression probe.

        WHY mean over tokens:
            Hallucination affects the whole completion, not a single token.
            Averaging over tokens produces a stable summary statistic and
            reduces noise from individual outlier positions (padding, special tokens).

        WHY 1 - cosine_sim:
            Cosine DISTANCE (1 - similarity) is bounded [0, 2].  Higher values
            indicate the layer representations are more different — more drift.
            Hallucinating text is expected to have higher drift than truthful text
            per arXiv 2604.13386 and arXiv 2601.14210.

        Args:
            hidden_states: Dict mapping layer index (int) to a 2-D numpy array
                           of shape (seq_len, hidden_dim).  Layer indices must
                           be keys present for all indices in self.layers.
                           When a required layer is absent, that pair's drift is 0.0.

        Returns:
            np.ndarray of shape (n_drift_pairs,), dtype float32.
            All values are clamped to [0, 2].

        Spec: REQ-TIER0-005-1
        """
        n = len(self.layers) - 1
        drift = np.zeros(n, dtype=np.float32)

        for i in range(n):
            layer_a = self.layers[i]
            layer_b = self.layers[i + 1]

            if layer_a not in hidden_states or layer_b not in hidden_states:
                # Missing layer → zero drift for this pair (no inflation).
                continue

            rep_a = hidden_states[layer_a]  # (seq_len, hidden_dim)
            rep_b = hidden_states[layer_b]  # (seq_len, hidden_dim)

            seq_len = min(rep_a.shape[0], rep_b.shape[0])
            if seq_len == 0:
                continue

            per_token = np.zeros(seq_len, dtype=np.float32)
            for t in range(seq_len):
                sim = _cosine_sim(rep_a[t], rep_b[t])
                per_token[t] = float(np.clip(1.0 - sim, 0.0, 2.0))

            drift[i] = float(np.mean(per_token))

        return drift

    def _run_model(self, text: str) -> dict[int, np.ndarray]:
        """Run model_runner and return hidden states, or empty dict on failure.

        Args:
            text: Input text to pass to model_runner.

        Returns:
            Dict[int, np.ndarray] of hidden states, or {} if model_runner is None
            or raises.
        """
        if self.model_runner is None:
            return {}
        try:
            return self.model_runner(text)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        correct_examples: list[str | dict],
        hallucinated_examples: list[str | dict],
    ) -> None:
        """Train the logistic regression probe on correct vs hallucinated examples.

        Extracts drift signatures for every example by calling model_runner,
        then fits a LogisticRegression classifier on the resulting feature matrix.

        When model_runner is None or unavailable, all signatures are zero vectors.
        LogisticRegression still fits (trivially), and predict_violation_prob will
        return 0.5 for all inputs — the correct behavior for CI runs without weights.

        WHY two separate lists instead of (text, label) tuples:
            Explicit separation makes it impossible to accidentally swap the label
            direction.  correct → label=0, hallucinated → label=1 is unambiguous.

        Args:
            correct_examples:     List of correct response strings (label=0),
                                  OR list of dicts with a "text" key.
            hallucinated_examples: List of hallucinated response strings (label=1),
                                   OR list of dicts with a "text" key.

        Spec: REQ-TIER0-005-2
        """
        from sklearn.linear_model import LogisticRegression

        def _get_text(ex):
            if isinstance(ex, dict):
                return ex.get("text", "")
            return str(ex)

        X_parts = []
        y_parts = []

        for ex in correct_examples:
            states = self._run_model(_get_text(ex))
            sig = self.extract_drift_signature(states)
            X_parts.append(sig)
            y_parts.append(0)

        for ex in hallucinated_examples:
            states = self._run_model(_get_text(ex))
            sig = self.extract_drift_signature(states)
            X_parts.append(sig)
            y_parts.append(1)

        X = np.vstack(X_parts).astype(np.float32)
        y = np.array(y_parts, dtype=int)

        self._probe = LogisticRegression(max_iter=500, random_state=42).fit(X, y)

    def fit_from_signatures(
        self,
        correct_sigs: np.ndarray,
        hallucinated_sigs: np.ndarray,
    ) -> None:
        """Train directly from precomputed drift signature matrices.

        Used when the caller has already extracted all drift signatures (e.g., to
        reuse extraction across multiple probe ablations without re-running the model).

        Args:
            correct_sigs:      Float32 array of shape (n_correct, n_drift_pairs).
            hallucinated_sigs: Float32 array of shape (n_halluc, n_drift_pairs).

        Spec: REQ-TIER0-005-2
        """
        from sklearn.linear_model import LogisticRegression

        X = np.vstack([correct_sigs, hallucinated_sigs]).astype(np.float32)
        y = np.array([0] * len(correct_sigs) + [1] * len(hallucinated_sigs), dtype=int)
        self._probe = LogisticRegression(max_iter=500, random_state=42).fit(X, y)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_violation_prob(self, hidden_states: dict[int, np.ndarray]) -> float:
        """Return probability that hidden_states correspond to a hallucinating response.

        Args:
            hidden_states: Dict[int, np.ndarray] as returned by model_runner.
                           Can also be the output of a synthetic generator.

        Returns:
            Float in [0, 1].  Higher = more likely hallucination.
            Returns 0.5 if the probe has not been fitted yet.

        Spec: REQ-TIER0-005-3
        """
        if self._probe is None:
            return 0.5

        sig = self.extract_drift_signature(hidden_states)
        classes = list(self._probe.classes_)
        halluc_class_idx = classes.index(1) if 1 in classes else 1
        proba = self._probe.predict_proba([sig])[0]
        return float(proba[halluc_class_idx])

    def predict_violation_prob_from_text(self, text: str) -> float:
        """Convenience: run model_runner then return violation probability.

        Args:
            text: Response text to score.

        Returns:
            Float in [0, 1].  Returns 0.5 if model_runner is None or probe not fitted.

        Spec: REQ-TIER0-005-3
        """
        states = self._run_model(text)
        return self.predict_violation_prob(states)
