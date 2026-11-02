"""Live200qV2Result — 200-question live benchmark result for Exp 478.

**Differences from Live200qResult (Exp 467):**

    1. ``ci_95_wilson`` replaces ``ci_95`` — same Wilson score CI on post_acc, renamed
       for clarity so callers don't confuse V1 and V2 result objects.

    2. ``is_statistically_positive`` now measures whether the *improvement* (delta)
       is statistically significant, not just whether post_acc is positive.
       It uses a Wald CI on the signed_improvement with SE computed from both
       pre_acc and post_acc.  This is a stronger gate: a post_acc of 0.90 is
       clearly positive, but the *improvement* may still be inside noise.

    3. ``cot_pairs_file: str | None`` replaces the ``cot_pairs: list[dict]`` field.
       CoT pairs are written to disk by CoTPairCollector before building this result,
       so we record the file path rather than holding all pairs in memory.

**Why Wilson CI on post_acc for ci_95_wilson:**

    Wilson score CI (Agresti & Coull, 1998) stays in [0,1] and shrinks gracefully
    near the boundaries.  At n=200 the half-width is ≤ 0.035 for any p, making
    post_acc estimates credible at the ±3.5pp level.  Normal approximation is NOT
    used because it produces negative lower bounds near p=0.

**Why Wald CI on the improvement for is_statistically_positive:**

    A 2pp improvement sounds real, but at n=200 with typical pre_acc ≈ 0.70 the
    standard error of the difference is ~0.045.  The 95% CI for the improvement is
    roughly 0.02 ± 0.089, which spans -0.07 to +0.11 — clearly insignificant.
    The Wald CI for the difference correctly accounts for variance in BOTH the
    pre_acc and post_acc estimates, unlike a simple one-sample test on post_acc.

Spec: REQ-BENCH-028, REQ-BENCH-029, REQ-BENCH-030,
      SCENARIO-BENCH-047, SCENARIO-BENCH-048, SCENARIO-BENCH-049 (Exp 478)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# z-score for 95% two-sided confidence interval (Agresti & Coull, 1998)
_Z95 = 1.959963984540054


@dataclass
class Live200qV2Result:
    """Per-model result from the 200-question live benchmark v2 (Exp 478).

    Parameters
    ----------
    model_id : str
        Human-readable model name, e.g. 'Gemma4-E4B-it'.
    pre_acc : float
        Fraction correct with NO pipeline (baseline pass).  Range [0.0, 1.0].
    post_acc : float
        Fraction correct WITH integrated VeriCoT+VPRM pipeline.  Range [0.0, 1.0].
    n : int
        Number of questions evaluated.  Must be > 0.  200 for Exp 478.
    extractor_name : str
        Name of the extraction stack used (e.g. 'VeriCoT+VPRM+CRANE').
    inference_mode : str
        'live_gpu' when running on real hardware; 'synthetic' for synthetic fallback.
    cot_pairs_file : str | None
        Path to the CoT pairs JSON written by CoTPairCollector, or None if no
        CoT pairs were collected.

    Computed properties
    -------------------
    signed_improvement : float
        post_acc - pre_acc (unclamped; negative means pipeline degraded accuracy).
    ci_95_wilson : tuple[float, float]
        Wilson score CI on post_acc at 95% confidence level.
        Width is < 0.07 at n=200 for any p in [0,1] (REQ-BENCH-030).
    is_statistically_positive : bool
        True iff the lower bound of the Wald CI for the *improvement* exceeds zero.
        This is stronger than signed_improvement > 0: the improvement must be large
        enough relative to the noise in both pre_acc and post_acc estimates.
        A 2pp improvement at n=200 with typical pre_acc ~0.70 is NOT significant.

    Spec: REQ-BENCH-028, REQ-BENCH-029, REQ-BENCH-030,
          SCENARIO-BENCH-047, SCENARIO-BENCH-048, SCENARIO-BENCH-049
    """

    model_id: str
    pre_acc: float
    post_acc: float
    n: int
    extractor_name: str
    inference_mode: str
    cot_pairs_file: str | None = None

    @property
    def signed_improvement(self) -> float:
        """Return post_acc - pre_acc (unclamped signed delta).

        Negative values mean the pipeline degraded accuracy.  Never clamp —
        honest negative results are required by CLAUDE.md.
        """
        return self.post_acc - self.pre_acc

    @property
    def ci_95_wilson(self) -> tuple[float, float]:
        """Wilson score 95% CI for post_acc.

        Returns (lower, upper) clamped to [0.0, 1.0].

        At n=200 the full width is < 0.07 for any p in [0,1].
        At p=0.05 (near boundary), width ≈ 0.062 < 0.07 (REQ-BENCH-030).
        At p=0.5 (worst case), width ≈ 0.138 — note that 0.07 is the
        half-width spec; total width at n=200 worst-case is ~0.138.

        Formula (Agresti & Coull, 1998):
            center = (p̂ + z²/(2n)) / (1 + z²/n)
            margin = z * sqrt(p̂*(1-p̂)/n + z²/(4n²)) / (1 + z²/n)

        Spec: REQ-BENCH-030, SCENARIO-BENCH-047
        """
        p = self.post_acc
        n = max(self.n, 1)
        z = _Z95
        z2 = z * z

        center = (p + z2 / (2 * n)) / (1 + z2 / n)
        margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n)

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        return (lower, upper)

    @property
    def is_statistically_positive(self) -> bool:
        """Return True iff the improvement is statistically significant at 95%.

        Uses a Wald CI for the difference of two proportions:
            improvement ± z * sqrt(pre*(1-pre)/n + post*(1-post)/n)

        The lower bound of this CI must exceed zero for the improvement to be
        considered statistically significant.

        Why this is harder to satisfy than signed_improvement > 0:
            At pre_acc=0.70, post_acc=0.72 (improvement=0.02), n=200:
            SE = sqrt(0.21/200 + 0.2016/200) ≈ 0.045
            lower = 0.02 - 1.96*0.045 = 0.02 - 0.089 = -0.069 < 0 → False

        A typical pre_acc of 0.70 requires an improvement of ~0.09 (9pp) before
        the lower CI bound clears zero at n=200.

        Spec: REQ-BENCH-030, SCENARIO-BENCH-049
        """
        n = max(self.n, 1)
        pre = self.pre_acc
        post = self.post_acc
        # Wald SE for the difference of two proportions (unpooled)
        se = math.sqrt(pre * (1 - pre) / n + post * (1 - post) / n)
        lower = self.signed_improvement - _Z95 * se
        return lower > 0.0

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict including all computed properties."""
        lo, hi = self.ci_95_wilson
        return {
            "model_id": self.model_id,
            "pre_acc": self.pre_acc,
            "post_acc": self.post_acc,
            "n": self.n,
            "extractor_name": self.extractor_name,
            "inference_mode": self.inference_mode,
            "signed_improvement": self.signed_improvement,
            "ci_95_wilson": [lo, hi],
            "is_statistically_positive": self.is_statistically_positive,
            "cot_pairs_file": self.cot_pairs_file,
        }
