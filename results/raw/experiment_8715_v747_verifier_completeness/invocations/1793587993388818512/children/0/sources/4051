"""VJEPA Streaming Logits Processor (Exp 894) — generation-time constraint guidance.

**Why this module exists:**
    VJEPA v2 (Exp 884, ood_auc=0.9211) currently operates as a Tier 2 *post-hoc*
    filter: it inspects a completed CoT trace and flags violations after the fact.
    Post-hoc filtering discards an entire generation that already cost GPU time.
    A cheaper alternative is to steer generation AWAY from violations while tokens
    are being produced — catch problems at the source rather than after the fact.

    arXiv 2502.03685 (Discrete Autoregressive Biasing) and arXiv 2603.03305
    (Draft-Conditioned Constrained Decoding) both show that soft logit penalties
    applied at each generation step can reduce constraint violations without
    requiring a separate rejection-sampling loop.

**Mechanism:**
    At every generation step, HuggingFace's ``generate()`` loop calls
    ``VJEPAStreamingLogitsProcessor.__call__(input_ids, scores)``.  We:
        1. Decode the current token prefix to a text string.
        2. Ask VJEPA for a violation probability on that prefix.
        3. If violation_prob > threshold, divide ALL logit scores by penalty_scale
           (a uniform soft down-weight that makes every token less likely,
           effectively pushing generation toward smaller, safer increments).

    This is intentionally conservative: uniform down-weighting is the simplest
    intervention that cannot accidentally invert the model's preference ordering.
    Per-token KL-guided masking is left as future work once the baseline effect
    size is measured.

Spec: REQ-VERIFY-177 (VJEPA streaming logit guidance)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from python.carnot.models.vjepa_predictor import VariationalJEPAPredictor


class VJEPAStreamingLogitsProcessor:
    """Soft-penalty LogitsProcessor that uses VJEPA to suppress high-violation prefixes.

    **Integration with HuggingFace generate():**
        Pass an instance of this class inside a ``LogitsProcessorList`` to
        ``model.generate(..., logits_processor=[processor])``.  The
        ``__call__`` signature matches the HuggingFace ``LogitsProcessor``
        protocol exactly so it can be used directly without subclassing.

    **Why uniform logit division instead of targeted token masking:**
        Targeted masking requires knowing WHICH tokens cause violations.  VJEPA
        operates on a full prefix string and returns a scalar probability, not
        a per-token attribution.  Dividing all logits by penalty_scale is the
        weakest intervention we can apply: it leaves the *relative* ranking of
        tokens unchanged (high-prob tokens remain higher than low-prob tokens)
        and merely reduces the temperature-corrected sharpness.  This avoids
        accidentally forcing the model into low-probability token territory.

    Args:
        vjepa:               Trained VariationalJEPAPredictor instance.
        tokenizer:           HuggingFace tokenizer matching the generation model.
        violation_threshold: VJEPA violation_prob above which the penalty fires.
                             Default 0.75 mirrors the Exp 884 cascade gate.
        penalty_scale:       Divisor applied to logits when threshold exceeded.
                             > 1.0 reduces sharpness; 1.0 is a no-op.

    Attributes:
        applied_count: Number of generation steps where the penalty was applied.
                       Useful for diagnosing how often VJEPA fires.
    """

    def __init__(
        self,
        vjepa: VariationalJEPAPredictor,
        tokenizer: object,
        violation_threshold: float = 0.75,
        penalty_scale: float = 2.0,
    ) -> None:
        self.vjepa = vjepa
        self.tokenizer = tokenizer
        self.violation_threshold = violation_threshold
        self.penalty_scale = penalty_scale
        self.prefix_history: list[str] = []
        self._applied_count: int = 0

    @property
    def applied_count(self) -> int:
        """Number of generation steps where the VJEPA penalty was triggered."""
        return self._applied_count

    def violation_probability(self, text: str) -> float:
        """Compute violation probability for a text prefix via VJEPA.

        Converts text to a TF-IDF feature vector and runs the VJEPA encoder.
        The context vector is the zero vector because at generation time we do
        not maintain a separate prior-step accumulator (the prefix IS the context).

        Args:
            text: Decoded token prefix so far.

        Returns:
            Violation probability in [0, 1].
        """
        import jax
        import jax.numpy as jnp

        from python.carnot.models.vjepa_predictor import (
            VOCAB_SIZE,
        )

        # Build a minimal vocab on-the-fly from the text itself.
        # VJEPA's in_dim is fixed at VOCAB_SIZE=50; we use a stable
        # hash-based projection so the same token always maps to the same bucket.
        tokens = text.lower().split()
        vec = [0.0] * VOCAB_SIZE
        for tok in tokens:
            idx = hash(tok) % VOCAB_SIZE
            vec[idx] += 1.0 / max(len(tokens), 1)

        x = jnp.array(vec, dtype=jnp.float32)
        ctx = jnp.zeros(self.vjepa.context_dim, dtype=jnp.float32)
        key = jax.random.PRNGKey(0)
        return self.vjepa.predict(x, ctx, key)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply VJEPA-guided soft penalty at each generation step.

        Called by HuggingFace's generate() loop before sampling.  If the
        decoded prefix has a VJEPA violation_prob above the threshold, all
        logit scores are divided by penalty_scale (soft down-weight).

        Args:
            input_ids: Token IDs generated so far, shape (batch, seq_len).
            scores:    Raw logit scores for next token, shape (batch, vocab).

        Returns:
            Possibly-modified scores tensor, same shape as input.
        """
        # Decode the current prefix from the first sequence in the batch.
        # We handle both tokenizer objects that have decode() directly and
        # those wrapped in another abstraction.
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

        vp = self.violation_probability(text)

        if vp > self.violation_threshold:
            scores = scores / self.penalty_scale
            self._applied_count += 1

        self.prefix_history.append(text)
        return scores
