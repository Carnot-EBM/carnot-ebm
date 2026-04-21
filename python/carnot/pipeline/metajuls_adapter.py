"""MetaJuLS-style online meta-RL adapter for LLMAsExtractorV1 policy parameters.

**Why this module exists (Self-Learning Tier 2 — Constraint Memory):**

    After each batch of live LLM outputs, the extractor's extraction quality
    can be measured by comparing detected violations against true labels.  A
    naive extractor keeps fixed hyperparameters (temperature, confidence
    threshold) forever.  MetaJuLS (arXiv 2601.00095) proposes a meta-RL loop
    that adapts the POLICY (the parameters that control how a downstream solver
    behaves) based on observed feedback — without full retraining.

    We apply this idea to LLMAsExtractorV1: after each batch, compute precision
    from (violation_detected, true_label) pairs, then nudge temperature and
    confidence_threshold in the direction that should improve future precision.

**Meta-RL update rules:**

    Low precision (< 0.5): extractor is flagging too many correct responses as
    violations.  Become more conservative: lower temperature (less creative
    extraction) and raise the confidence bar (require stronger evidence).

    High precision (> 0.8): extractor is being cautious and precise.  Reward
    by relaxing constraints slightly to improve recall headroom.

    Neither: hold current policy — no update needed for this batch.

**Tier 2 self-learning (Constraint Memory) positioning:**

    Tier 1 (ConstraintAdditionFromMemory) adds new constraint TERMS when a
    failure pattern recurs.  Tier 2 (this module) adjusts the POLICY that
    governs how the extractor itself behaves.  Both tiers can run in parallel.

Spec: REQ-LEARN-078, REQ-LEARN-079,
      SCENARIO-LEARN-121, SCENARIO-LEARN-122, SCENARIO-LEARN-123
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# ExtractorPolicy — the mutable state that MetaJuLSAdapter updates
# ---------------------------------------------------------------------------


@dataclass
class ExtractorPolicy:
    """Hyperparameters that govern LLMAsExtractorV1 extraction behaviour.

    These are the "policy parameters" in MetaJuLS terminology: the knobs that
    a meta-RL agent adjusts based on downstream feedback signals (precision,
    recall) rather than gradient descent on a loss.

    Fields:
        temperature               — LLM sampling temperature for JsonClaimExtractor
                                    and SymCodeExtractor prompts.  Lower = more
                                    deterministic / conservative extraction.
        claim_confidence_threshold — Minimum confidence to count an extracted
                                    claim as a real violation signal.  Higher =
                                    fewer but more reliable detections.
        strategy_weights          — Relative weight for each extraction strategy
                                    when blending results.  Keys: 'json', 'symcode',
                                    'chain'.  Values are unnormalised positive floats.
    """

    temperature: float = 0.1
    claim_confidence_threshold: float = 0.5
    strategy_weights: dict[str, float] = field(
        default_factory=lambda: {"json": 0.33, "symcode": 0.33, "chain": 0.34}
    )

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON artifact embedding."""
        return {
            "temperature": self.temperature,
            "claim_confidence_threshold": self.claim_confidence_threshold,
            "strategy_weights": dict(self.strategy_weights),
        }


# ---------------------------------------------------------------------------
# MetaJuLSAdapter — online policy update from batch feedback
# ---------------------------------------------------------------------------


class MetaJuLSAdapter:
    """Online meta-RL adapter that updates ExtractorPolicy from live batch results.

    **How adaptation works (one batch at a time):**

        1. Receive a list of (response, violation_detected, true_label) triples
           representing one batch of live model outputs.
        2. Compute precision = TP / (TP + FP + epsilon).
           - TP: the extractor said "violation" and the response IS incorrect.
           - FP: the extractor said "violation" but the response was correct.
        3. Apply the meta-RL rule to update self.policy (see module docstring).
        4. Append the batch precision to self.experience for trend tracking.

    **Thread safety:** not thread-safe.  Call from a single thread or wrap
    with an external lock when used in a parallel pipeline.

    Args:
        initial_policy : Optional starting policy.  If None, uses default
                         ExtractorPolicy with equal strategy weights.

    Spec: REQ-LEARN-078, REQ-LEARN-079
    """

    def __init__(self, initial_policy: ExtractorPolicy | None = None) -> None:
        self.policy: ExtractorPolicy = initial_policy or ExtractorPolicy(
            strategy_weights={"json": 0.33, "symcode": 0.33, "chain": 0.34}
        )
        # Each entry: {'batch_id': int, 'precision': float}
        # Grows monotonically — never trimmed — so offline replay is possible.
        self.experience: list[dict] = []

    def update_from_batch(self, batch_results: list[dict]) -> ExtractorPolicy:
        """Compute batch precision and apply meta-RL policy update.

        Parameters
        ----------
        batch_results : list of dicts, each with keys:
            'response'           — raw LLM response text (unused in update, for traceability)
            'violation_detected' — bool: did the extractor flag this response?
            'true_label'         — 'correct' | 'incorrect': ground truth label

        Why 'correct'/'incorrect' strings instead of booleans:
            The live corpus stores labels as human-readable strings ('correct',
            'incorrect').  Using the string form avoids a double-inversion bug
            where callers confuse is_correct=True with violation_detected=True.

        Returns the updated ExtractorPolicy (also mutates self.policy in place).

        Spec: REQ-LEARN-078, SCENARIO-LEARN-121, SCENARIO-LEARN-122
        """
        # TP: extractor fired AND the response truly had an error (incorrect).
        tp = sum(
            1
            for r in batch_results
            if r["violation_detected"] and r["true_label"] != "correct"
        )
        # FP: extractor fired AND the response was actually correct.
        fp = sum(
            1
            for r in batch_results
            if r["violation_detected"] and r["true_label"] == "correct"
        )
        # Epsilon guard prevents division by zero when no violations were detected.
        precision = tp / (tp + fp + 1e-9)

        # Meta-RL update: low precision → be more conservative.
        if precision < 0.5:
            self.policy.temperature = max(0.01, self.policy.temperature * 0.9)
            self.policy.claim_confidence_threshold = min(
                0.9, self.policy.claim_confidence_threshold * 1.1
            )
        elif precision > 0.8:
            # High precision → we can afford to relax and improve recall headroom.
            self.policy.temperature = min(0.5, self.policy.temperature * 1.05)
            self.policy.claim_confidence_threshold = max(
                0.3, self.policy.claim_confidence_threshold * 0.95
            )
        # Else: precision in [0.5, 0.8] — policy is performing acceptably, hold steady.

        self.experience.append(
            {"batch_id": len(self.experience), "precision": precision}
        )
        return self.policy

    def precision_trend(self) -> float:
        """Slope of precision over the last 3 batches.  Positive = improving.

        Why look back only 3 batches:
            Recent performance matters more than historical performance in an
            online setting.  A 3-batch window is wide enough to smooth noise
            but short enough to reflect recent drift quickly.

        Returns 0.0 when fewer than 2 batches have been observed (undefined
        trend).

        Spec: REQ-LEARN-079, SCENARIO-LEARN-123
        """
        if len(self.experience) < 2:
            return 0.0
        # Use the last 3 entries (or fewer if < 3 batches recorded).
        recent = self.experience[-3:]
        precisions = [e["precision"] for e in recent]
        # Trend = last precision minus first precision in the window.
        # Positive: precision improved.  Negative: precision degraded.
        return precisions[-1] - precisions[0]
