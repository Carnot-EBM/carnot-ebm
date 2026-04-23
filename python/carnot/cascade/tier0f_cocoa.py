"""Tier 0f CoCoA — training-free inter-layer disagreement hallucination detector.

WHY THIS MODULE EXISTS (arXiv 2602.09486 "Listen to the Layers: Mitigating Hallucinations
with Inter-Layer Disagreement", March 2026):
    Standard hallucination detectors require either fine-tuned probes (expensive, needs
    labels) or post-hoc factual verification (requires external knowledge).  CoCoA
    proposes a third option: measure how much the model's internal representation of
    a claim *changes* from early to late transformer layers.  When the model is
    "confused" — uncertain about a factual claim — the hidden states at different
    depths disagree sharply.  When the model is confident, the representation
    stabilises across layers.

    The Contrastive Multi-Layer Disagreement Score (ConMLDS) formalises this:
        ConMLDS = mean over (early, late) pairs of: 1 - cosine_similarity(h_early, h_late)
    where h_l is the hidden state at the last input token from transformer layer l.
    High ConMLDS => representational instability => likely hallucination.

WHY THIS IS TIER 0f (NOT A HARD GATE):
    arXiv 2602.09486 reports AUC ~0.65-0.72 on TruthfulQA and FEVER benchmarks.
    This is useful signal but not reliable enough to short-circuit the cascade.
    Wiring it as an *advisory* Tier 0f signal lets downstream probes incorporate
    the inter-layer disagreement without risking false rejections.

    The energy-domain analogy: ConMLDS measures energy *gradient* across the depth
    dimension of the transformer.  High gradient = the model is still "searching"
    for a stable representation.  Low gradient = the model converged to a confident
    answer.  This is orthogonal to Tier 0a-0e energy magnitudes, which measure
    constraint satisfaction rather than representational stability.

HOW IT REUSES EXISTING INFRASTRUCTURE:
    JEPAReasonerProbe (Exp 730/732) already provides:
      - load_model(): loads Qwen3.5-0.8B with output_hidden_states=True
      - extract_hidden_state(text): single-layer extraction at last token
    CoCoADetector reuses the same model object (pass by reference) to avoid loading
    Qwen3.5-0.8B twice.  The only new work is extracting at multiple layers and
    computing pairwise cosine distances.

CALIBRATION:
    Threshold is calibrated as mean(conmlds_correct) + 1.0 * std(conmlds_correct)
    on the FoVer v2 correct examples.  This means 84% of correct examples fall
    below the threshold (1-sigma rule) and is_unstable=True fires only for the
    most divergent 16% of correct examples plus a larger fraction of hallucinations.

Spec: REQ-VERIFY-151, REQ-VERIFY-152, SCENARIO-VERIFY-201, SCENARIO-VERIFY-202
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class CoCoADetector:
    """Training-free inter-layer disagreement scorer (arXiv 2602.09486).

    Given a loaded LLM, this class computes ConMLDS by comparing hidden states
    at early transformer layers vs late transformer layers for the last input token.
    No fine-tuning or probe weights are needed — this is purely forward-pass geometry.

    Parameters
    ----------
    model : object
        A loaded transformers AutoModelForCausalLM with output_hidden_states=True.
        Typically the same model object held by JEPAReasonerProbe — pass it in to
        avoid double-loading Qwen3.5-0.8B (~1.5 GB VRAM).
    tokenizer : object
        The corresponding AutoTokenizer.  Must match the model's vocabulary.
    early_layers : tuple of int
        Layer indices considered "early" in the transformer stack.  For Qwen3.5-0.8B
        (28 total layers), (8, 10, 12) captures the lower-middle representations
        where basic semantic information has been assembled but reasoning is still
        in progress.
    late_layers : tuple of int
        Layer indices considered "late".  (14, 16) is the upper-middle band where
        reasoning outputs crystallise.  Comparing these to early layers captures
        the representational "journey" for a factual claim.
    threshold : float | None
        ConMLDS above which is_unstable=True.  When None, score() returns
        is_unstable=None until calibrate() has been called.
    device : str
        PyTorch device string.  Defaults to "cuda:0" when GPU is available.

    Spec: REQ-VERIFY-151
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        early_layers: tuple[int, ...] = (8, 10, 12),
        late_layers: tuple[int, ...] = (14, 16),
        threshold: Optional[float] = None,
        device: str = "cpu",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.early_layers = early_layers
        self.late_layers = late_layers
        self.threshold = threshold
        self.device = device

    # ------------------------------------------------------------------
    # Hidden-state extraction  (REQ-VERIFY-151-1)
    # ------------------------------------------------------------------

    def extract_hidden_states(
        self, text: str, layer_indices: list[int]
    ) -> dict[int, np.ndarray]:
        """Run a single forward pass and extract last-token hidden states at requested layers.

        WHY a single forward pass for all layers: output_hidden_states=True returns
        ALL intermediate hidden states in one shot — there is no need to re-run the
        forward pass per layer.  We run once and index into the tuple.

        WHY last token: in a causal LM the last input token has attended to all prior
        tokens, making its hidden state a summary of the full prefix up to that point.
        arXiv 2602.09486 §3.1 confirms that the last-token position yields the highest
        CoCoA discriminability across all benchmarks tested.

        Parameters
        ----------
        text : str
            The prompt text whose representations we want to interrogate.
        layer_indices : list of int
            Which transformer layer indices to extract (0-indexed, where 0 = first block).

        Returns
        -------
        dict mapping layer_index (int) -> np.ndarray of shape (hidden_dim,)
            Float32 last-token hidden state for each requested layer.

        Spec: REQ-VERIFY-151-1
        """
        import torch  # noqa: PLC0415

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of length (n_layers + 1).
        # Index 0 = embedding output. Index k+1 = output of transformer block k.
        # So layer index L corresponds to hidden_states[L + 1].
        result: dict[int, np.ndarray] = {}
        for li in layer_indices:
            hs = outputs.hidden_states[li + 1]  # (1, seq_len, hidden_dim)
            result[li] = hs[0, -1, :].cpu().float().numpy()  # (hidden_dim,)
        return result

    # ------------------------------------------------------------------
    # ConMLDS computation  (REQ-VERIFY-151-2)
    # ------------------------------------------------------------------

    def compute_conmlds(
        self,
        early_state: np.ndarray,
        late_state: np.ndarray,
    ) -> float:
        """Compute cosine distance between an early and late hidden state.

        ConMLDS for a single pair = 1 - cosine_similarity(early, late).

        WHY cosine distance rather than L2: the absolute magnitude of hidden states
        varies across layers (later layers tend to have larger norms after LayerNorm).
        Cosine distance is scale-invariant, so it measures directional shift regardless
        of magnitude — exactly the "representational instability" we want to capture.

        Parameters
        ----------
        early_state : np.ndarray
            Hidden state vector from an early layer.  Shape (hidden_dim,).
        late_state : np.ndarray
            Hidden state vector from a later layer.  Same shape.

        Returns
        -------
        float
            Cosine distance in [0, 1].  0 = identical direction, 1 = orthogonal.

        Spec: REQ-VERIFY-151-2
        """
        # Compute cosine similarity: dot(a, b) / (|a| * |b|)
        # Use clipping to avoid tiny numerical negatives from float32 precision.
        dot = float(np.dot(early_state, late_state))
        norm_e = float(np.linalg.norm(early_state))
        norm_l = float(np.linalg.norm(late_state))

        # Guard against zero-norm vectors (e.g., padding-only inputs).
        if norm_e < 1e-10 or norm_l < 1e-10:
            return 0.0

        cos_sim = np.clip(dot / (norm_e * norm_l), -1.0, 1.0)
        return float(1.0 - cos_sim)

    # ------------------------------------------------------------------
    # Full scoring  (REQ-VERIFY-151)
    # ------------------------------------------------------------------

    def score(self, text: str) -> tuple[float, Optional[bool]]:
        """Compute ConMLDS for text and return (conmlds, is_unstable).

        Extracts hidden states at all early and late layers in a single forward pass,
        then averages the cosine distance over all (early, late) pairs.

        Parameters
        ----------
        text : str
            The prompt text to score.

        Returns
        -------
        tuple[float, bool | None]
            - conmlds: mean cosine distance over all (early, late) layer pairs.
            - is_unstable: True iff conmlds > self.threshold.  None if threshold
              is not calibrated yet.

        Spec: REQ-VERIFY-151, REQ-VERIFY-151-4
        """
        all_layers = list(self.early_layers) + list(self.late_layers)
        states = self.extract_hidden_states(text, all_layers)

        # Compute mean cosine distance over all (early, late) pairs.
        distances: list[float] = []
        for el in self.early_layers:
            for ll in self.late_layers:
                distances.append(self.compute_conmlds(states[el], states[ll]))

        conmlds = float(np.mean(distances)) if distances else 0.0

        is_unstable: Optional[bool] = None
        if self.threshold is not None:
            is_unstable = bool(conmlds > self.threshold)

        return conmlds, is_unstable

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score_batch(self, texts: list[str]) -> list[tuple[float, Optional[bool]]]:
        """Score a list of texts sequentially, returning (conmlds, is_unstable) per text.

        WHY sequential rather than batched forward pass: each text may have a different
        sequence length, which makes padding tricky for hidden-state extraction at the
        *last* token (the last real token, not the padding token).  Sequential extraction
        is simpler and the GPU is still well-utilised because each forward pass is large
        enough to saturate the compute units.
        """
        return [self.score(t) for t in texts]

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------

    def calibrate(self, correct_texts: list[str]) -> dict[str, float]:
        """Calibrate the is_unstable threshold from correct (non-hallucinated) texts.

        Sets threshold = mean(conmlds_correct) + 1.0 * std(conmlds_correct).
        At this 1-sigma level, ~84% of correct texts fall below the threshold —
        meaning is_unstable fires on only the top 16% of correct texts (false
        positive rate ~16%) plus a larger fraction of actual hallucinations.

        This is intentionally conservative: a higher threshold would miss more
        hallucinations, and a lower threshold would fire too often on correct text
        and become useless as an advisory signal.

        Parameters
        ----------
        correct_texts : list of str
            Texts that are known to be correct (non-hallucinated) responses.
            Used to estimate the "baseline" ConMLDS distribution.

        Returns
        -------
        dict with keys: mean_conmlds, std_conmlds, threshold
        """
        scores = [self.score(t)[0] for t in correct_texts]
        arr = np.array(scores, dtype=np.float32)
        mean_c = float(arr.mean())
        std_c = float(arr.std())
        self.threshold = mean_c + 1.0 * std_c
        return {
            "mean_conmlds": mean_c,
            "std_conmlds": std_c,
            "threshold": self.threshold,
        }
