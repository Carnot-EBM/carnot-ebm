"""Precision100qResult — 100-question benchmark result with Wilson 95% CI.

**Why this extends LivePrecisionResult (Exp 451 → Exp 464):**

    LivePrecisionResult stores signed_improvement and is_positive, which are
    correct but provide no confidence interval.  At 50 questions a 5pp improvement
    has a ±14pp Wilson CI, which overlaps zero — the claim is not statistically
    credible.  At 100 questions the same improvement has a ±10pp CI, narrow enough
    to make directional claims credible.

    Precision100qResult adds:
    - n_questions : int — actual sample count (100 for Exp 464)
    - extractor_used : str — which extractors produced violations (for RETRO-033 tracing)
    - inference_mode : str — 'live_gpu' or 'synthetic' (for audit trail)
    - confidence_interval_95 : tuple[float, float] — Wilson score interval on post_accuracy

**Why Wilson score over normal approximation (Clopper-Pearson alternative):**

    The normal approximation CI (p ± 1.96*sqrt(p*(1-p)/n)) breaks down near 0% and 100%:
    it can produce intervals like [-0.02, 0.08] which contain negative probabilities.
    Wilson score shrinks gracefully at extremes and is always within [0,1].  It is the
    standard CI for proportions in benchmarking literature (Agresti & Coull, 1998).

    Formula: ( p̂ + z²/(2n) ± z*sqrt(p̂*(1-p̂)/n + z²/(4n²)) ) / (1 + z²/n)
    where z=1.96 for 95% CI.

Spec: REQ-BENCH-014, SCENARIO-BENCH-034
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# z-score for 95% two-sided confidence interval
_Z95 = 1.959963984540054


@dataclass
class Precision100qResult:
    """Per-model result from the 100-question live precision benchmark (Exp 464).

    Extends the LivePrecisionResult concept (Exp 451) with a question count,
    extractor identification, inference mode, and Wilson 95% confidence interval.

    Parameters
    ----------
    model_id : str
        Human-readable model name, e.g. 'Gemma4-E4B-it' or 'Qwen3.5-0.8B'.
    pre_accuracy : float
        Fraction correct with NO pipeline applied (baseline).  Range [0.0, 1.0].
    post_accuracy : float
        Fraction correct WITH pipeline applied.  Range [0.0, 1.0].
    n_questions : int
        Number of questions in the benchmark run.  Must be > 0.
    extractor_used : str
        Comma-separated names of extractors that produced violations (e.g.
        'vericot,vprm').  'none' when no violations were found.
    inference_mode : str
        'live_gpu' when running on real hardware; 'synthetic' when using synthetic
        questions (no real GPU available).

    Computed properties
    -------------------
    signed_improvement : float
        post_accuracy - pre_accuracy (unclamped signed delta).
    confidence_interval_95 : tuple[float, float]
        Wilson score interval on post_accuracy at 95% confidence level.
    is_positive : bool
        True iff signed_improvement > 0 (strict).

    Spec: REQ-BENCH-014, SCENARIO-BENCH-034
    """

    model_id: str
    pre_accuracy: float
    post_accuracy: float
    n_questions: int
    extractor_used: str
    inference_mode: str

    @property
    def signed_improvement(self) -> float:
        """Return post_accuracy - pre_accuracy (unclamped).

        Negative values mean the pipeline made things worse — never clamp this.
        """
        return self.post_accuracy - self.pre_accuracy

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """Wilson score 95% CI for post_accuracy.

        Returns (lower, upper) clamped to [0.0, 1.0].

        Why Wilson score:
            The normal approximation breaks down at p ≈ 0 or p ≈ 1.  Wilson score
            always produces a valid probability interval and is standard in NLP
            benchmarking literature for proportion-based accuracy metrics.

        Formula (Agresti & Coull, 1998):
            center = (p̂ + z²/(2n)) / (1 + z²/n)
            margin = z * sqrt(p̂*(1-p̂)/n + z²/(4n²)) / (1 + z²/n)

        Spec: REQ-BENCH-014, SCENARIO-BENCH-034
        """
        p = self.post_accuracy
        n = max(self.n_questions, 1)  # guard against n=0
        z = _Z95
        z2 = z * z

        center = (p + z2 / (2 * n)) / (1 + z2 / n)
        margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n)

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        return (lower, upper)

    @property
    def is_positive(self) -> bool:
        """Return True iff signed_improvement > 0 (strict greater-than).

        Zero improvement is NOT a positive result.
        """
        return self.signed_improvement > 0

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict including all computed properties."""
        lo, hi = self.confidence_interval_95
        return {
            "model_id": self.model_id,
            "pre_accuracy": self.pre_accuracy,
            "post_accuracy": self.post_accuracy,
            "n_questions": self.n_questions,
            "extractor_used": self.extractor_used,
            "inference_mode": self.inference_mode,
            "signed_improvement": self.signed_improvement,
            "confidence_interval_95": [lo, hi],
            "is_positive": self.is_positive,
        }
