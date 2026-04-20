"""MISE dense reward calibration for EORM training signal.

**Researcher summary (REQ-LEARN-070, arXiv 2604.11611):**
    MISE (Mutual Information Signal Estimator) backward inference asks: given a
    model response, what constraint was the response *trying* to satisfy?  If
    the model succeeded, the response should be semantically aligned with the
    question that prompted it.

    We operationalise alignment as cosine similarity between embeddings of the
    response and the question.  For (original, repair, verdict_correct) triples
    from live verify-repair runs:

      - Correct responses (verdict_correct=True) should have HIGH similarity to
        the originating question — the answer is actually about the question.
      - Incorrect responses (verdict_correct=False) often drift off-topic, so
        similarity should be LOWER.

    The *calibration gap* (mean_alignment_correct - mean_alignment_incorrect)
    is positive when the MISE signal reliably separates correct from incorrect
    outputs.  A positive gap means we can use embedding cosine similarity as a
    dense reward signal to calibrate EORM training without human labels.

**Why backward inference matters:**
    Standard EORM reward is binary (pass/fail per question).  Backward inference
    provides a dense signal even within the "pass" bucket — responses that are
    highly aligned with the question are better constrained than responses that
    just happen to produce the right final number.  This is the dense reward
    calibration described in arXiv 2604.11611 §4.

Spec: REQ-LEARN-070, REQ-LEARN-071, SCENARIO-LEARN-110, SCENARIO-LEARN-111,
      SCENARIO-LEARN-112
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class MISETriple:
    """One (question, response, verdict) record from a live verify-repair run.

    **Why this exists:**
        MISE calibration needs triples to compute alignment statistics.  Each
        triple links a question to the model response it produced (either the
        original or the repaired response) plus a boolean verdict from the
        verifier.  Storing the original_response separately allows offline
        analysis of repair quality even when repair was not attempted.

    Fields
    ------
    question
        The original prompt or question text.
    original_response
        The model's first response before any repair attempt.
    repaired_response
        The repaired response text, or None when no repair was attempted.
    verdict_correct
        True when the verifier accepted the final response as correct.

    Spec: REQ-LEARN-070, SCENARIO-LEARN-110
    """

    question: str
    original_response: str
    repaired_response: str | None
    verdict_correct: bool


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class MISECalibrator:
    """Calibrate EORM reward using MISE backward inference alignment.

    **Why cosine similarity as the proxy:**
        arXiv 2604.11611 §4 defines backward inference probability as
        p(question | response).  Approximating this with embedding cosine
        similarity is the natural tractable proxy: high cosine similarity means
        the response is semantically close to the question space, which
        correlates with the model having internalised the constraint the question
        encodes.

    **Limitation acknowledged:**
        The default embed_fn used in experiments is a deterministic hash
        projection (``lambda x: jnp.array([hash(x) % 128])``).  This is a
        placeholder — it preserves the MISE pipeline structure without requiring
        a neural encoder at experiment time.  Production use would replace this
        with a real sentence encoder (e.g. BGE-M3, multilingual-e5).

    Parameters
    ----------
    embed_fn
        A callable that takes a string and returns a 1-D array-like of floats.
        The array must support standard arithmetic (addition, multiplication)
        so that cosine similarity can be computed.  At experiment time this is
        typically a JAX array; at test time it can be a plain Python list.

    Spec: REQ-LEARN-070, REQ-LEARN-071, SCENARIO-LEARN-110, SCENARIO-LEARN-111,
          SCENARIO-LEARN-112
    """

    def __init__(self, embed_fn: Callable[[str], object]) -> None:
        self._embed_fn = embed_fn

    # ------------------------------------------------------------------
    # Core alignment computation
    # ------------------------------------------------------------------

    def backward_inference_score(self, response: str, question: str) -> float:
        """Return cosine similarity between embed(response) and embed(question).

        This is the MISE backward inference proxy: how much does the response
        "look like" it was trying to answer the question?  Higher = more
        aligned.

        The implementation avoids a division-by-zero by returning 0.0 when
        either embedding has zero norm.

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

        Spec: REQ-LEARN-071, SCENARIO-LEARN-111
        """
        r_vec = self._embed_fn(response)
        q_vec = self._embed_fn(question)

        # Compute dot product and norms using standard Python arithmetic so this
        # works with both plain lists and JAX arrays.
        def _dot(a: object, b: object) -> float:
            # Support both sequences and array-like objects with element access.
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
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, triples: list[MISETriple]) -> dict[str, float]:
        """Compute MISE calibration statistics over a list of triples.

        For each triple the "effective response" is the repaired_response when
        available, otherwise the original_response.  The alignment score is
        computed between the effective response and the question.

        Returns a dict with:
          - ``mean_alignment_correct``: mean cosine similarity for triples where
            verdict_correct=True.
          - ``mean_alignment_incorrect``: mean cosine similarity for triples
            where verdict_correct=False.
          - ``calibration_gap``: mean_alignment_correct - mean_alignment_incorrect.
            Positive means the MISE signal separates correct from incorrect.

        When all triples share the same verdict, the missing group is scored
        as 0.0 (no data to estimate from).

        Parameters
        ----------
        triples
            List of MISETriple records.  May be empty — returns all-zero stats.

        Returns
        -------
        dict[str, float]
            Keys: 'mean_alignment_correct', 'mean_alignment_incorrect',
            'calibration_gap'.

        Spec: REQ-LEARN-070, SCENARIO-LEARN-110, SCENARIO-LEARN-112
        """
        correct_scores: list[float] = []
        incorrect_scores: list[float] = []

        for triple in triples:
            effective = triple.repaired_response if triple.repaired_response is not None else triple.original_response
            score = self.backward_inference_score(effective, triple.question)
            if triple.verdict_correct:
                correct_scores.append(score)
            else:
                incorrect_scores.append(score)

        mean_correct = sum(correct_scores) / len(correct_scores) if correct_scores else 0.0
        mean_incorrect = sum(incorrect_scores) / len(incorrect_scores) if incorrect_scores else 0.0

        return {
            "mean_alignment_correct": mean_correct,
            "mean_alignment_incorrect": mean_incorrect,
            "calibration_gap": mean_correct - mean_incorrect,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "MISETriple",
    "MISECalibrator",
]
