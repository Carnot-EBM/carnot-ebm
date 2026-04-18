"""LivePrecisionResult — signed-improvement data class for Exp 451 post-fix benchmark.

**Why this module exists (RETRO-028 follow-up):**
    Exp 439 measured 0% Gemma4 accuracy due to the llama.cpp tokenizer bug (RETRO-028).
    Exp 451 re-runs the benchmark with GemmaTransformersLoader (the fix) and compares
    pre_accuracy (BASELINE variant, no pipeline) against post_accuracy (pipeline variant).

    ``LivePrecisionResult`` encapsulates exactly this comparison: two floats and two
    derived signals — signed_improvement and is_positive.  The design mirrors
    MicroPrecisionResult from precision_micro.py but is scoped to the single-variant
    post-fix comparison (pre vs. post) rather than a multi-variant ablation.

**Why ``signed_improvement`` is never clamped:**
    Negative values are honest regression signals.  If the pipeline makes things worse,
    the caller must see that negative number — clamping to zero hides the regression.
    This invariant is consistent with MicroPrecisionResult from Exp 439.

**Why ``is_positive`` is a strict > 0 check:**
    A zero improvement (post_accuracy == pre_accuracy) is NOT a positive result.
    The pipeline must produce measurable gain to claim success.

Spec: REQ-BENCH-013, SCENARIO-BENCH-031, SCENARIO-BENCH-032
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LivePrecisionResult:
    """Per-model result from the live precision post-fix benchmark (Exp 451).

    Stores baseline accuracy (no pipeline) alongside pipeline accuracy and derives
    the signed improvement and positivity flag as computed properties.

    Parameters
    ----------
    model_id : str
        Human-readable model name, e.g. ``'Gemma4-E4B-it'`` or ``'Qwen3.5-0.8B'``.
        Used as a key in the artifact dict for downstream correlation.
    pre_accuracy : float
        Baseline accuracy: fraction of questions answered correctly with NO
        verify-repair pipeline applied.  Expected range [0.0, 1.0].
    post_accuracy : float
        Pipeline accuracy: fraction of questions answered correctly WITH the
        verify-repair pipeline applied.  Expected range [0.0, 1.0].

    Computed properties
    -------------------
    signed_improvement : float
        ``post_accuracy - pre_accuracy``.  Positive = improvement; negative = regression.
        NEVER clamped — callers see the honest signed delta.
    is_positive : bool
        ``True`` iff ``signed_improvement > 0``.  Zero improvement is NOT positive.

    Spec: REQ-BENCH-013, SCENARIO-BENCH-031
    """

    model_id: str
    pre_accuracy: float
    post_accuracy: float

    @property
    def signed_improvement(self) -> float:
        """Return post_accuracy - pre_accuracy (unclamped signed delta).

        Negative values are honest regression signals — never abs() or clamp this.

        Spec: REQ-BENCH-013
        """
        return self.post_accuracy - self.pre_accuracy

    @property
    def is_positive(self) -> bool:
        """Return True iff signed_improvement is strictly greater than zero.

        Zero improvement is explicitly NOT positive — the pipeline must produce
        measurable gain to claim success.

        Spec: REQ-BENCH-013, SCENARIO-BENCH-031
        """
        return self.signed_improvement > 0

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict of all fields including computed properties.

        Used by experiment artifact builders to embed per-model results.
        """
        return {
            "model_id": self.model_id,
            "pre_accuracy": self.pre_accuracy,
            "post_accuracy": self.post_accuracy,
            "signed_improvement": self.signed_improvement,
            "is_positive": self.is_positive,
        }
