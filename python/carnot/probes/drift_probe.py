"""DRIFTProbe — Tier 0i hidden-state representational drift hallucination probe.

**Researcher summary:**
    arXiv 2601.14210 (DRIFT) shows that hallucinating LLMs exhibit representational
    drift between adjacent Transformer layers: cosine similarity between layer L and
    layer L+1 hidden states is lower for hallucinating completions than for truthful
    ones.  This white-box probe requires access to internal hidden states, which is
    architecturally different from Carnot's existing logit-space or attention-space probes
    (Tier 0a-0h).

    This module implements Tier 0i (DRIFTProbe variant) as an advisory probe:
        1. Extract hidden states from a small LLM (Qwen/Qwen3.5-0.8B) at
           selected intermediate layers.
        2. Compute drift signatures: per-layer-pair cosine distance between
           consecutive layer representations, averaged over token positions.
        3. Train a logistic regression probe on labeled (text, label) pairs
           to distinguish hallucinating from truthful completions.
        4. At inference time, predict hallucination probability from the drift
           signature vector.

    WHY a small model (Qwen3.5-0.8B)?
        The full SOTA GGUFs (Qwen3.6-35B-A3B, Gemma-4-31B) are too large for
        CPU-only hidden-state extraction in a reasonable time budget.  The 0.8B
        model runs in ~4 s/text on CPU (no CUDA required), making it viable for
        training a probe on 60 FoVer pairs within the experiment time budget.
        The probe is a LINEAR classifier on top of the drift signatures — the
        classifier quality depends on the discriminability of drift patterns, not
        on the model's output quality, so a smaller model is acceptable here.

    WHY layers [4, 8, 12, 16]?
        The DRIFT paper reports that middle-to-late layer transitions exhibit the
        largest drift gap between truthful and hallucinating completions.  Layers
        4→8, 8→12, and 12→16 span the model's middle layers (Qwen3.5-0.8B has
        24 transformer blocks), catching the representational shift without
        including early embedding layers (noisy) or the final layers (logit-dominated).

    CI safety:
        When transformers is not installed or the model cannot be loaded,
        extract_drift_signature() returns a zero vector of the correct shape.
        This lets tests run without downloading any model weights.

Spec: REQ-TIER0-009, SCENARIO-TIER0-009
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression as _LR


def _cosine_similarity_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Returns the dot product divided by the product of norms.  When either
    vector is all-zeros (e.g., a padding token), returns 1.0 (no drift) to
    avoid division by zero and prevent padding tokens from inflating the drift
    estimate.

    Args:
        a: 1-D float array, shape (D,).
        b: 1-D float array, shape (D,).

    Returns:
        Cosine similarity in [-1, 1].  Returns 1.0 for zero vectors.
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class DRIFTProbe:
    """Tier 0i representational-drift hallucination probe.

    **For engineers:**
        This probe extracts hidden-state representations from an LLM at selected
        layer indices, computes cosine-distance drift between consecutive layers,
        and trains a logistic regression classifier on those drift signatures to
        distinguish hallucinating from truthful text.

        The probe operates as an advisory flag: it adds `is_representationally_drifted`
        to the verify_extended() result dict but does NOT change the verified outcome.
        This matches the advisory pattern used by Tier 0g (StreamingCoTHalluDetector)
        and the original Tier 0i (HalluSAEGeometricProbe).

    Usage:
        probe = DRIFTProbe(model_name="Qwen/Qwen3.5-0.8B", probe_layers=[4, 8, 12, 16])
        probe.fit(train_pairs)   # [{text: str, label: int (1=hallucinating, 0=truthful)}]
        flag = probe.is_representationally_drifted("The answer is ...")

    Args:
        model_name:   HuggingFace model ID to load for hidden-state extraction.
                      Must be loadable locally (no internet required in CI).
        probe_layers: Transformer layer indices at which hidden states are extracted.
                      Drift is computed for each consecutive pair, yielding
                      len(probe_layers)-1 values in the signature vector.
                      Default [4, 8, 12, 16] → 3-value signature.

    Spec: REQ-TIER0-009
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        probe_layers: list[int] | None = None,
    ) -> None:
        self.model_name = model_name
        self.probe_layers = probe_layers if probe_layers is not None else [4, 8, 12, 16]
        # Linear probe trained by fit(); None until fit() is called.
        self.linear_probe: _LR | None = None
        # Cached model and tokenizer — loaded lazily on first extract call.
        self._model = None
        self._tokenizer = None
        self._model_load_failed = False

    def _load_model(self) -> bool:
        """Lazy-load the LLM for hidden-state extraction.

        Returns True if model is loaded and ready, False if load failed or
        transformers is not installed.  On failure, sets _model_load_failed=True
        so subsequent calls are no-ops (fast path).

        WHY lazy loading:
            Importing transformers and loading a 0.8B model takes ~3-5 s.  Deferring
            this to the first extract_drift_signature() call ensures that tests that
            use mock pairs (synthetic drift signatures) are not penalized.
        """
        if self._model is not None:
            return True
        if self._model_load_failed:
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=False
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    output_hidden_states=True,
                    trust_remote_code=False,
                    torch_dtype=torch.float32,
                )
                self._model.eval()
            return True
        except Exception:
            self._model_load_failed = True
            return False

    def extract_drift_signature(self, text: str) -> np.ndarray:
        """Extract the drift signature for a text string.

        Runs a forward pass through the LLM, collects hidden states at
        self.probe_layers, and computes mean-cosine-distance drift between
        each consecutive pair of selected layers.

        WHY mean over token positions:
            Taking the mean over all token positions collapses the per-token
            drift tensor to a single scalar per layer-pair.  Tokens in the same
            response show correlated drift patterns (a hallucination tends to
            affect the whole completion), so the mean is a stable summary statistic.

        Drift clamping:
            drift = 1.0 - cosine_similarity.  Because cosine_similarity ∈ [-1, 1],
            raw drift ∈ [0, 2].  We clamp to [0, 2] explicitly to document the
            valid range.  Negative cosine_similarity (anti-aligned representations)
            produces drift > 1.0, which is unusual but valid (rare in practice for
            adjacent layers).

        CI safety:
            When the model is not loadable (no transformers, no weights),
            returns np.zeros(len(self.probe_layers) - 1) so callers always
            get an array of the correct shape.

        Args:
            text: Input text to compute the drift signature for.

        Returns:
            np.ndarray of shape (len(self.probe_layers) - 1,) containing
            mean-over-tokens cosine drift for each consecutive layer pair.
            Values are clamped to [0, 2].

        Spec: REQ-TIER0-009-1
        """
        n_pairs = len(self.probe_layers) - 1
        zero_signature = np.zeros(n_pairs, dtype=np.float32)

        if not self._load_model():
            return zero_signature

        try:
            import torch

            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)

            # hidden_states is a tuple of tensors: (n_layers+1, batch, seq_len, hidden_dim)
            # Index 0 is the embedding layer; index L corresponds to transformer block L.
            hidden_states = outputs.hidden_states  # tuple

            # Check we have enough layers
            max_layer = max(self.probe_layers)
            if max_layer >= len(hidden_states):
                return zero_signature

            # Build per-layer mean representation: shape (n_tokens, hidden_dim) -> (hidden_dim,)
            # Using the mean over tokens gives a single representative vector per layer.
            layer_reps = {}
            for layer_idx in self.probe_layers:
                # hidden_states[layer_idx] shape: (1, seq_len, hidden_dim)
                hs = hidden_states[layer_idx][0].numpy()  # (seq_len, hidden_dim)
                layer_reps[layer_idx] = hs

            # Compute per-token drift for each consecutive pair, then take mean.
            drift_values = np.zeros(n_pairs, dtype=np.float32)
            for i in range(n_pairs):
                layer_a = self.probe_layers[i]
                layer_b = self.probe_layers[i + 1]
                rep_a = layer_reps[layer_a]  # (seq_len, hidden_dim)
                rep_b = layer_reps[layer_b]  # (seq_len, hidden_dim)

                seq_len = rep_a.shape[0]
                per_token_drift = np.zeros(seq_len, dtype=np.float32)
                for t in range(seq_len):
                    cossim = _cosine_similarity_vectors(rep_a[t], rep_b[t])
                    # drift = 1 - cosine_similarity, clamped to [0, 2]
                    per_token_drift[t] = float(np.clip(1.0 - cossim, 0.0, 2.0))

                drift_values[i] = float(np.mean(per_token_drift))

            return drift_values

        except Exception:
            return zero_signature

    def fit(self, pairs: list[dict]) -> None:
        """Train the linear logistic regression probe on labeled text pairs.

        Extracts drift signatures for all texts in pairs, then fits a
        LogisticRegression classifier.  After this call, predict_proba()
        and is_representationally_drifted() are available.

        WHY logistic regression:
            The DRIFT paper uses a linear probe (logistic regression) to evaluate
            whether drift signatures are linearly separable between hallucinating
            and truthful completions.  A linear probe is preferred because it is
            interpretable (the coefficients tell us which layer-pair drift matters
            most) and has minimal overfitting risk on small training sets (50 pairs).

        Args:
            pairs: List of dicts with keys:
                   - "text": str — the LLM response text to score
                   - "label": int — 1 for hallucinating, 0 for truthful

        Spec: REQ-TIER0-009-2
        """
        from sklearn.linear_model import LogisticRegression

        X = np.vstack([self.extract_drift_signature(p["text"]) for p in pairs])
        y = np.array([int(p["label"]) for p in pairs])

        # Handle the degenerate case where model load failed and all signatures are zero:
        # LogisticRegression will still fit (all features identical) but predict_proba
        # will return 0.5 for everything.  This is the correct behavior for CI runs
        # where no model weights are present.
        self.linear_probe = LogisticRegression(max_iter=200, random_state=42).fit(X, y)

    def predict_proba(self, text: str) -> float:
        """Return the probability that text is representationally drifted (hallucinating).

        Args:
            text: The LLM response text to evaluate.

        Returns:
            Float in [0, 1].  Higher values indicate more likely hallucination.
            Returns 0.5 if the probe has not been fitted yet.

        Spec: REQ-TIER0-009-3
        """
        if self.linear_probe is None:
            return 0.5
        sig = self.extract_drift_signature(text)
        return float(self.linear_probe.predict_proba([sig])[0][1])

    def is_representationally_drifted(
        self, text: str, threshold: float = 0.6
    ) -> bool:
        """Return True when the text's drift signature exceeds the hallucination threshold.

        Advisory flag — callers should NOT use this to block verification;
        it is metadata for downstream decision surfaces and logging.

        Args:
            text:      The LLM response text.
            threshold: Probability threshold above which text is flagged as drifted.
                       Default 0.6 (conservative; lower recall, fewer false positives).

        Returns:
            True if predict_proba(text) > threshold, False otherwise.

        Spec: REQ-TIER0-009-4
        """
        return self.predict_proba(text) > threshold
