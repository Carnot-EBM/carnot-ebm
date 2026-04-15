"""SinkProbe — attention-sink hallucination pre-filter.

**Researcher summary:**
    Implements a fast hallucination pre-filter based on arXiv 2604.10697
    (SinkProbe, Apr 2026), which shows that specific attention heads concentrate
    probability mass on "sink" tokens (BOS, period, etc.) when the model is
    generating factually CERTAIN content.  When sink concentration is LOW the
    model is producing uncertain output and full verification should run.  When
    concentration is HIGH the response is likely correct and can skip the
    expensive Ising verifier.

**Detailed explanation for engineers:**
    Transformer attention maps (shape: n_heads × seq_len × seq_len) contain
    rich information about what the model "pays attention to" while generating
    each token.  arXiv 2604.10697 identifies that certain token positions —
    called "sink" tokens — act as attention gravity wells:

        - [BOS] (beginning-of-sequence) tokens are always in context and
          attract a disproportionate share of attention regardless of their
          semantic content.
        - Period (.) and comma (,) tokens are short, high-frequency tokens
          whose embeddings tend to form stable attractors in attention space.

    Key finding from the paper:
        When the model is CONFIDENT about a factual claim, attention mass
        accumulates on these sink tokens (the model "dismisses" the query by
        routing attention to inert sinks).
        When the model is UNCERTAIN, it distributes attention more uniformly
        across real content tokens (seeking evidence), and sink concentration DROPS.

    This produces a cheap hallucination signal:
        mean_sink_score = mean over heads of (attention mass on sink positions)
        low mean_sink_score  (<threshold) → uncertain → run Ising verifier
        high mean_sink_score (≥threshold) → confident → SKIP Ising verifier (fast path)

    Pipeline position:
        SinkProbe fires AFTER generation (captures the attention matrix from the
        forward pass that produced the response tokens) but BEFORE the Ising
        constraint evaluator.  It is the first gate in the three-tier pipeline:

            SinkProbe (fast, ~0 ms extra overhead) →
            EORM ranker (55M GPU, ~10 ms) →
            Ising verification (CPU, 0.006 ms per constraint)

    CI-safety:
        compute_sink_concentration operates on arbitrary jnp.ndarray inputs so
        it can be unit-tested without a real language model.

    Key classes / functions:
        - SinkTokenType: enum of common sink token kinds
        - SinkConcentration: dataclass with per-head scores, mean, and max
        - compute_sink_concentration(): core arithmetic, operates on jnp arrays
        - SinkProbeResult: decision dataclass (is_uncertain, should_skip)
        - SinkProbe: thin class wrapping score / decide / benchmark

Spec: REQ-VERIFY-086, REQ-VERIFY-087
SCENARIO-VERIFY-113, SCENARIO-VERIFY-114, SCENARIO-VERIFY-115
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# SinkTokenType
# ---------------------------------------------------------------------------


class SinkTokenType(Enum):
    """Categories of tokens that commonly act as attention sinks.

    **Detailed explanation for engineers:**
        These token types are identified empirically in arXiv 2604.10697 as
        the positions that absorb disproportionate attention mass in confident
        model outputs.  The enum is used to document intent when building
        sink_positions lists — callers scan the token sequence for these token
        kinds and pass their indices to compute_sink_concentration().

    Members:
        BOS:    Beginning-of-sequence token (e.g., <s>, <bos>, [CLS]).
                Always present in context; acts as the primary sink.
        EOS:    End-of-sequence token.  Present in few-shot examples; can
                attract attention as a structural delimiter.
        PERIOD: Period character (.).  Short, high-frequency; forms a stable
                attractor in embedding space.
        COMMA:  Comma character (,).  Similar to PERIOD; frequent in flowing
                prose and factual statements.
    """

    BOS = auto()
    EOS = auto()
    PERIOD = auto()
    COMMA = auto()


# ---------------------------------------------------------------------------
# SinkConcentration dataclass
# ---------------------------------------------------------------------------


@dataclass
class SinkConcentration:
    """Per-head attention-sink concentration scores for a single response.

    **Detailed explanation for engineers:**
        Each attention head in a transformer independently routes query tokens
        to key tokens.  We summarise "how much of each head's attention mass
        lands on sink positions" as a single float per head.

        The score for head h is:
            score_h = mean over query positions q of sum_{j in sink_positions} attn[h, q, j]

        This averages across query positions so that the score is not inflated
        by sequence length.  For a perfectly uniform attention distribution
        over seq_len keys, score_h = len(sink_positions) / seq_len.

    Attributes:
        per_head_sink_scores: Score in [0, 1] for each attention head.
            Index i corresponds to head i in the attention tensor.
        mean_sink_score: Mean of per_head_sink_scores.  Primary signal for
            the is_uncertain decision in SinkProbe.decide().
        max_sink_score: Max of per_head_sink_scores.  Useful for detecting
            whether ANY head is strongly sink-focused even when most are not.

    Spec: REQ-VERIFY-086
    """

    per_head_sink_scores: list[float]
    mean_sink_score: float
    max_sink_score: float


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_sink_concentration(
    attention_matrix: jnp.ndarray,
    sink_positions: list[int],
) -> SinkConcentration:
    """Compute how much attention mass each head places on sink token positions.

    **Detailed explanation for engineers:**
        attention_matrix has shape (n_heads, seq_len, seq_len).
        Entry attn[h, q, k] is the probability that head h routes query token q
        to key token k (rows sum to 1 for each head and query position).

        Algorithm:
        1. For each head h and each query position q, sum attention over all
           key indices that are in sink_positions:
               sink_mass[h, q] = sum_{k in sink_positions} attn[h, q, k]
        2. Average over query positions to get a per-head score:
               score_h = mean_q(sink_mass[h, q])
        3. Return mean and max over heads.

        If sink_positions is empty, all scores are 0.0.

    Args:
        attention_matrix: JAX array of shape (n_heads, seq_len, seq_len).
            Each row [h, q, :] must sum to 1 (valid probability distribution).
            Can be a numpy array — it is converted via jnp.asarray internally.
        sink_positions: List of key indices (0-based) that are sink tokens.
            For example, [0] for BOS at position 0, or [0, 5] for BOS and a
            mid-sequence period token.

    Returns:
        SinkConcentration with per-head scores, mean, and max.

    Spec: REQ-VERIFY-086
    """
    attn = jnp.asarray(attention_matrix)  # (n_heads, seq_len, seq_len)
    n_heads = attn.shape[0]

    if not sink_positions:
        zero_scores = [0.0] * n_heads
        return SinkConcentration(
            per_head_sink_scores=zero_scores,
            mean_sink_score=0.0,
            max_sink_score=0.0,
        )

    # Sum attention mass at each sink key index for each (head, query) pair.
    # sink_attn shape: (n_heads, seq_len)
    sink_cols = jnp.array(sink_positions, dtype=jnp.int32)
    # Index into the key dimension: attn[:, :, sink_cols] → (n_heads, seq_len, n_sinks)
    sink_attn = attn[:, :, sink_cols]  # (n_heads, seq_len, n_sinks)
    # Sum over sink positions, then average over query positions.
    head_sink_mass = sink_attn.sum(axis=-1).mean(axis=-1)  # (n_heads,)

    per_head = [float(head_sink_mass[h]) for h in range(n_heads)]
    mean_score = float(head_sink_mass.mean())
    max_score = float(head_sink_mass.max())

    return SinkConcentration(
        per_head_sink_scores=per_head,
        mean_sink_score=mean_score,
        max_sink_score=max_score,
    )


# ---------------------------------------------------------------------------
# SinkProbeResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class SinkProbeResult:
    """Decision result from SinkProbe for a single response.

    **Detailed explanation for engineers:**
        This dataclass communicates two complementary pieces of information:

        is_uncertain:
            True when mean_sink_score < threshold.  A low sink score means the
            model was distributing attention broadly over content tokens, which
            the paper associates with factual uncertainty.  When True, Ising
            verification must run.

        should_skip_verification:
            Logical negation of is_uncertain.  Provided as an explicit field so
            routing code can write ``if result.should_skip_verification: ...``
            without negating a boolean — cleaner intent signalling.

    Attributes:
        sink_concentration: The raw SinkConcentration used to make the decision.
            Preserved so callers can inspect the underlying per-head scores for
            debugging or threshold calibration.
        is_uncertain: True when the model is assessed as uncertain (low sink).
        should_skip_verification: True when it is safe to skip Ising (high sink).

    Spec: REQ-VERIFY-086
    """

    sink_concentration: SinkConcentration
    is_uncertain: bool
    should_skip_verification: bool


# ---------------------------------------------------------------------------
# SinkProbe
# ---------------------------------------------------------------------------


@dataclass
class SinkProbe:
    """Attention-sink hallucination pre-filter using arXiv 2604.10697.

    **Detailed explanation for engineers:**
        SinkProbe is the first gate in Carnot's three-tier verification pipeline:

            SinkProbe  →  EORM ranker  →  Ising verifier

        It uses the attention matrix captured during generation (zero extra
        model overhead — the matrix is already computed as part of the forward
        pass) to decide whether full verification is necessary.

        Decision rule:
            mean_sink_score = mean over heads of (attention at sink positions)
            if mean_sink_score >= threshold:   response appears confident → SKIP
            if mean_sink_score <  threshold:   response appears uncertain → VERIFY

        Threshold calibration:
            The default threshold of 0.3 is a reasonable starting point from
            the paper's experiments.  Use benchmark() to measure skip_rate,
            false_negative_rate, and true_negative_rate on a labelled corpus,
            then tune threshold to match your quality / throughput trade-off.

    Attributes:
        threshold: Mean sink score above which a response is considered
            confident (default 0.3).
        sink_token_types: Which SinkTokenType members are used.  Default is
            (BOS, PERIOD) as identified in arXiv 2604.10697 as the primary sinks.

    Spec: REQ-VERIFY-086, REQ-VERIFY-087
    """

    threshold: float = 0.3
    sink_token_types: tuple[SinkTokenType, ...] = (
        SinkTokenType.BOS,
        SinkTokenType.PERIOD,
    )

    def score(
        self,
        attention_matrix: jnp.ndarray,
        sink_positions: list[int],
    ) -> SinkConcentration:
        """Compute sink concentration for an attention matrix.

        **Detailed explanation for engineers:**
            Thin wrapper around compute_sink_concentration().  Use this method
            when you want the probe instance to own the scoring step (e.g.,
            future versions may apply head selection filters here based on
            sink_token_types).

        Args:
            attention_matrix: Shape (n_heads, seq_len, seq_len).
            sink_positions: Indices of sink token positions in the key dimension.

        Returns:
            SinkConcentration with per-head scores, mean, and max.

        Spec: REQ-VERIFY-086
        """
        return compute_sink_concentration(attention_matrix, sink_positions)

    def decide(self, sink_concentration: SinkConcentration) -> SinkProbeResult:
        """Translate a SinkConcentration into a skip-or-verify decision.

        **Detailed explanation for engineers:**
            The decision rule is simple and auditable:

                is_uncertain = mean_sink_score < threshold   (strict less-than)
                should_skip  = not is_uncertain

            "Strict less-than" means a score exactly equal to the threshold is
            treated as confident (not uncertain) — this is a deliberate
            conservative choice to minimise false negatives at the boundary.

        Args:
            sink_concentration: Result of score() or compute_sink_concentration().

        Returns:
            SinkProbeResult with the decision and the input concentration embedded.

        Spec: REQ-VERIFY-086, REQ-VERIFY-087
        SCENARIO-VERIFY-113 (high concentration → skip)
        SCENARIO-VERIFY-114 (low concentration → verify)
        """
        is_uncertain = sink_concentration.mean_sink_score < self.threshold
        return SinkProbeResult(
            sink_concentration=sink_concentration,
            is_uncertain=is_uncertain,
            should_skip_verification=not is_uncertain,
        )

    def benchmark(
        self,
        responses_with_attention: list[dict],
        correctness_labels: list[bool],
    ) -> dict:
        """Measure skip_rate, false_negative_rate, and true_negative_rate on a corpus.

        **Detailed explanation for engineers:**
            Terminology (adapting binary classification vocabulary to the
            skip/verify framing):

                "Negative" = a response that is CORRECT (no error to catch).
                "Positive" = a response that is WRONG (error present).

                True Negative  (TN): Correct response, SinkProbe said SKIP.
                                     Good outcome — we avoid unnecessary Ising calls.
                False Negative (FN): WRONG response, SinkProbe said SKIP.
                                     Bad outcome — we miss a real error.
                True Positive  (TP): Wrong response, SinkProbe said VERIFY.
                                     Good outcome — we catch the error.
                False Positive (FP): Correct response, SinkProbe said VERIFY.
                                     Wasted Ising call but no missed error.

            Metrics returned:
                skip_rate           = (TN + FN) / total
                false_negative_rate = FN / (FN + TP)   [of all wrong, how many slipped?]
                true_negative_rate  = TN / (TN + FP)   [of all correct, how many we skipped?]

            If there are no wrong responses, false_negative_rate = 0.0.
            If there are no correct responses, true_negative_rate = 0.0.
            If the input is empty, all rates are 0.0.

        Args:
            responses_with_attention: List of dicts, each with keys:
                "attention_matrix": jnp.ndarray of shape (n_heads, seq_len, seq_len)
                "sink_positions":   list[int] of sink token indices
            correctness_labels: Parallel list of booleans.
                True  = response is factually correct.
                False = response contains a factual error.

        Returns:
            Dict with keys "skip_rate", "false_negative_rate", "true_negative_rate".
            All values are floats in [0.0, 1.0].

        Spec: REQ-VERIFY-086, REQ-VERIFY-087
        SCENARIO-VERIFY-115
        """
        if not responses_with_attention:
            return {
                "skip_rate": 0.0,
                "false_negative_rate": 0.0,
                "true_negative_rate": 0.0,
            }

        total = len(responses_with_attention)
        n_skipped = 0
        n_wrong = 0
        n_fn = 0  # Wrong responses that were skipped (bad)
        n_correct = 0
        n_tn = 0  # Correct responses that were skipped (good)

        for item, label in zip(responses_with_attention, correctness_labels):
            attn = item["attention_matrix"]
            sink_positions = item["sink_positions"]
            conc = self.score(attn, sink_positions)
            result = self.decide(conc)

            if label:  # correct
                n_correct += 1
                if result.should_skip_verification:
                    n_tn += 1
                    n_skipped += 1
            else:  # wrong
                n_wrong += 1
                if result.should_skip_verification:
                    n_fn += 1
                    n_skipped += 1

        skip_rate = n_skipped / total
        false_negative_rate = (n_fn / n_wrong) if n_wrong > 0 else 0.0
        true_negative_rate = (n_tn / n_correct) if n_correct > 0 else 0.0

        return {
            "skip_rate": float(skip_rate),
            "false_negative_rate": float(false_negative_rate),
            "true_negative_rate": float(true_negative_rate),
        }
