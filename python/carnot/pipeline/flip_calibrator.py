"""FLIP backward inference reward calibration for repair quality scoring.

**Researcher summary (REQ-LEARN-076, arXiv 2602.13551):**
    FLIP (Forward-Looking Inference Probe) extends the MISE backward inference
    idea to REPAIR quality specifically.  Given a repaired response, backward
    inference asks: "what constraint was this response satisfying?"  If the
    repair IMPROVED constraint alignment (higher cosine similarity to the
    original question), we treat that as a positive reward signal.  If the
    repair REDUCED alignment, the repair may have introduced a constraint-
    inconsistent change — FLIP detects it.

    The key difference from MISE (arXiv 2604.11611):
      - MISE measures alignment across BOTH correct and incorrect responses to
        compute a calibration gap.
      - FLIP focuses on the DELTA between original and repaired responses to
        detect whether the repair moved in the right direction.  This makes it
        a direct repair quality signal, not just a response quality signal.

    A positive FLIP delta (flip_score > original_score) means the repair
    brought the response closer to the question's constraint space.  A negative
    delta means the repair drifted away — a sign of constraint-inconsistent
    edits.

**Implementation note on embed_fn:**
    The default embed_fn used in experiments is a deterministic hash projection
    (``lambda x: jnp.array([hash(w)%128 for w in x.split()[:128]])``) that
    preserves pipeline structure without requiring a neural encoder at
    experiment time.  Production use would replace this with a real sentence
    encoder (e.g., BGE-M3).

Spec: REQ-LEARN-076, REQ-LEARN-077,
      SCENARIO-LEARN-118, SCENARIO-LEARN-119, SCENARIO-LEARN-120
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FLIPRepairTriple:
    """One (question, original, repaired, verdict) record for FLIP calibration.

    **Why this exists:**
        FLIP calibration needs to compare the constraint alignment of the
        ORIGINAL response against the REPAIRED response.  Storing both lets us
        compute the alignment delta: did the repair move the response closer to
        or further from the constraint the question encodes?

    Fields
    ------
    question
        The original prompt or question text.
    original
        The model's response before any repair attempt.
    repaired
        The repaired response text, or None when no repair was attempted.
        When None, flip_score is also None and the triple is skipped in
        batch_calibrate statistics.
    verdict_correct
        True when the verifier accepted the final response as correct.
    flip_score
        Set by FLIPRewardCalibrator.calibrate_repair().  The cosine similarity
        between embed(repaired) and embed(question).  Higher = more aligned.
        None until calibrate_repair() is called.

    Spec: REQ-LEARN-076, SCENARIO-LEARN-118
    """

    question: str
    original: str
    repaired: str | None
    verdict_correct: bool
    flip_score: float | None = field(default=None)


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class FLIPRewardCalibrator:
    """Calibrate repair quality using FLIP backward inference alignment delta.

    **Why backward inference as a repair quality signal:**
        Standard verify-repair pipelines report a binary verdict: did the final
        response pass the verifier?  This binary signal is weak because it
        does not distinguish between repairs that legitimately corrected the
        response and repairs that incidentally passed the verifier while
        introducing constraint-inconsistent changes (e.g., changing a number to
        match the answer key without fixing the underlying reasoning step).

        FLIP (arXiv 2602.13551) detects the second case: a repair that passes
        the verifier but REDUCES cosine similarity to the question has likely
        changed the response in a constraint-inconsistent direction.

    **How the alignment delta works:**
        For a triple (question Q, original response O, repaired response R):
          - original_score = cosine_similarity(embed(O), embed(Q))
          - flip_score     = cosine_similarity(embed(R), embed(Q))
          - delta          = flip_score - original_score

        Positive delta: repair moved R closer to Q's constraint space — good.
        Negative delta: repair moved R away from Q's constraint space — suspect.
        Zero delta: repair had no effect on alignment — neutral.

    Parameters
    ----------
    embed_fn
        A callable that takes a string and returns a 1-D array-like of floats.
        The array must support element-wise iteration so cosine similarity can
        be computed.  At experiment time this is typically a JAX array; at test
        time it can be a plain Python list.

    Spec: REQ-LEARN-076, REQ-LEARN-077, SCENARIO-LEARN-118, SCENARIO-LEARN-119,
          SCENARIO-LEARN-120
    """

    def __init__(self, embed_fn: Callable[[str], object]) -> None:
        self._embed_fn = embed_fn

    # ------------------------------------------------------------------
    # Core alignment computation  (identical contract to MISECalibrator)
    # ------------------------------------------------------------------

    def backward_inference_score(self, response: str, question: str) -> float:
        """Return cosine similarity between embed(response) and embed(question).

        Higher values mean the response is more semantically aligned with the
        question — i.e., more likely to have been generated by a model that
        internalised the question's constraint.

        Avoids division-by-zero: returns 0.0 when either embedding is all-zero.

        Parameters
        ----------
        response
            The response text to score.
        question
            The question that prompted the response.

        Returns
        -------
        float
            Cosine similarity in [-1.0, 1.0].  Returns 0.0 for zero-norm inputs.

        Spec: REQ-LEARN-076, SCENARIO-LEARN-119
        """
        r_vec = self._embed_fn(response)
        q_vec = self._embed_fn(question)

        def _dot(a: object, b: object) -> float:
            try:
                return float(sum(float(ai) * float(bi) for ai, bi in zip(a, b)))  # type: ignore[call-overload]
            except TypeError:
                return float(a) * float(b)  # type: ignore[arg-type]

        def _norm(a: object) -> float:
            try:
                return float(sum(float(ai) ** 2 for ai in a)) ** 0.5  # type: ignore[call-overload]
            except TypeError:
                return abs(float(a))  # type: ignore[arg-type]

        dot = _dot(r_vec, q_vec)
        norm_r = _norm(r_vec)
        norm_q = _norm(q_vec)

        if norm_r == 0.0 or norm_q == 0.0:
            return 0.0

        return dot / (norm_r * norm_q)

    # ------------------------------------------------------------------
    # Per-triple calibration
    # ------------------------------------------------------------------

    def calibrate_repair(self, triple: FLIPRepairTriple) -> FLIPRepairTriple:
        """Set triple.flip_score based on FLIP backward inference delta.

        If repaired is not None, computes:
          - original_score = backward_inference_score(original, question)
          - flip_score     = backward_inference_score(repaired, question)

        Sets triple.flip_score = flip_score when flip_score > original_score
        (repair improved alignment), otherwise sets it to flip_score anyway
        so batch_calibrate can see the delta direction.

        When repaired is None, sets triple.flip_score = None (no repair to
        evaluate).

        Parameters
        ----------
        triple
            A FLIPRepairTriple.  Mutated in-place and also returned.

        Returns
        -------
        FLIPRepairTriple
            The same triple with flip_score set.

        Spec: REQ-LEARN-076, SCENARIO-LEARN-118
        """
        if triple.repaired is None:
            triple.flip_score = None
            return triple

        triple.flip_score = self.backward_inference_score(triple.repaired, triple.question)
        return triple

    # ------------------------------------------------------------------
    # Batch calibration
    # ------------------------------------------------------------------

    def batch_calibrate(self, triples: list[FLIPRepairTriple]) -> dict:
        """Compute FLIP calibration statistics over a list of repair triples.

        For each triple that has a non-None repaired response, computes the
        FLIP alignment delta and accumulates:
          - mean_flip_score: average flip_score across triples with repairs.
          - n_improved: count of triples where flip_score > original alignment.
          - repair_quality: 'good' when n_improved > n_evaluated/2 (majority
            of repairs improved alignment), 'bad' when n_improved == 0 and
            n_evaluated > 0, otherwise 'neutral'.

        Triples with repaired=None are skipped — they contribute 0 to all
        statistics.  An empty triples list returns all-zero / 'neutral'.

        Parameters
        ----------
        triples
            List of FLIPRepairTriple records.

        Returns
        -------
        dict
            Keys: 'mean_flip_score' (float), 'n_improved' (int),
            'repair_quality' ('good' | 'neutral' | 'bad').

        Spec: REQ-LEARN-076, SCENARIO-LEARN-120
        """
        flip_scores: list[float] = []
        n_improved = 0

        for triple in triples:
            if triple.repaired is None:
                continue
            original_score = self.backward_inference_score(triple.original, triple.question)
            calibrated = self.calibrate_repair(triple)
            flip_score = calibrated.flip_score
            if flip_score is None:
                continue
            flip_scores.append(flip_score)
            if flip_score > original_score:
                n_improved += 1

        n_evaluated = len(flip_scores)
        mean_flip_score = sum(flip_scores) / n_evaluated if n_evaluated > 0 else 0.0

        if n_evaluated == 0:
            repair_quality = "neutral"
        elif n_improved > n_evaluated / 2:
            repair_quality = "good"
        elif n_improved == 0:
            repair_quality = "bad"
        else:
            repair_quality = "neutral"

        return {
            "mean_flip_score": mean_flip_score,
            "n_improved": n_improved,
            "repair_quality": repair_quality,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FLIPRepairTriple",
    "FLIPRewardCalibrator",
]
