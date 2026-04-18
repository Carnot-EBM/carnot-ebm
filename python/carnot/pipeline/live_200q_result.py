"""Live200qResult — 200-question live benchmark result with Wilson 95% CI.

**Why 200 questions (vs 100 in Exp 464):**

    At n=100 a 5pp improvement has a ±10pp Wilson CI, which overlaps zero for
    modest improvements.  At n=200 the same improvement has a ±7pp CI, and the
    half-width shrinks to ≤3.5pp even near the boundary (p≈0.05), making
    directional claims statistically credible.  This is the minimum sample size
    recommended by Agresti & Coull for benchmarking proportions.

**Why Wilson score CI over normal approximation:**

    The normal approximation (p ± 1.96*sqrt(p*(1-p)/n)) can produce negative
    lower bounds when p is near 0.  Wilson score always stays in [0,1] and
    shrinks gracefully at extremes.  It is the standard for NLP benchmarking.

**cot_pairs field:**

    Each item is a dict with keys: model, question, cot_text, correct.
    These are written to results/exp467_cot_pairs.json for Exp 472 JEPA retrain.
    An empty list is valid (no violations found, no pairs collected).

Spec: REQ-BENCH-017, REQ-BENCH-018, REQ-BENCH-019,
      SCENARIO-BENCH-036, SCENARIO-BENCH-037, SCENARIO-BENCH-038
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# z-score for 95% two-sided confidence interval (Agresti & Coull, 1998)
_Z95 = 1.959963984540054


@dataclass
class Live200qResult:
    """Per-model result from the 200-question live integrated benchmark (Exp 467).

    Extends Precision100qResult (Exp 464) with:
    - is_statistically_positive: True iff the lower CI bound exceeds zero.
      This is a stronger claim than is_positive (which is just delta > 0).
    - cot_pairs: list of reasoning traces collected for Exp 472 JEPA retrain.

    Parameters
    ----------
    model_id : str
        Human-readable model name, e.g. 'Gemma4-E4B-it'.
    pre_acc : float
        Fraction correct with NO pipeline (baseline).  Range [0.0, 1.0].
    post_acc : float
        Fraction correct WITH integrated VeriCoT+VPRM pipeline.  Range [0.0, 1.0].
    n : int
        Number of questions evaluated.  Must be > 0.  200 for Exp 467.
    extractor_name : str
        Name of the extraction stack used (e.g. 'VeriCoT+VPRM+CRANE').
    inference_mode : str
        'live_gpu' when running on real hardware; 'synthetic' when using the
        synthetic question fallback (e.g. datasets package unavailable).

    Computed properties
    -------------------
    signed_improvement : float
        post_acc - pre_acc (unclamped; negative means pipeline degraded accuracy).
    ci_95 : tuple[float, float]
        Wilson score interval on post_acc at 95% confidence level.
        Half-width is ≤ 0.035 when n >= 200 (REQ-BENCH-019).
    is_statistically_positive : bool
        True iff the lower bound of ci_95 > 0.
        A stronger claim than signed_improvement > 0: even the pessimistic end
        of the confidence interval rules out zero improvement.
    cot_pairs : list[dict]
        Reasoning traces collected during pipeline pass, for JEPA retrain.

    Spec: REQ-BENCH-017, REQ-BENCH-018, REQ-BENCH-019
    """

    model_id: str
    pre_acc: float
    post_acc: float
    n: int
    extractor_name: str
    inference_mode: str
    cot_pairs: list[dict] = field(default_factory=list)

    @property
    def signed_improvement(self) -> float:
        """Return post_acc - pre_acc (unclamped signed delta).

        Negative values mean the pipeline degraded accuracy.  Never clamp —
        honest negative results are required by CLAUDE.md ('The energy function
        is ground truth').
        """
        return self.post_acc - self.pre_acc

    @property
    def ci_95(self) -> tuple[float, float]:
        """Wilson score 95% CI for post_acc.

        Returns (lower, upper) clamped to [0.0, 1.0].

        At n=200 the half-width is ≤ 0.035 for any p in [0,1], satisfying
        REQ-BENCH-019.  At p=0.5 (worst case) the half-width is ≈0.069/2 ≈ 0.034.

        Formula (Agresti & Coull, 1998):
            center = (p̂ + z²/(2n)) / (1 + z²/n)
            margin = z * sqrt(p̂*(1-p̂)/n + z²/(4n²)) / (1 + z²/n)

        Spec: REQ-BENCH-019, SCENARIO-BENCH-036
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
        """Return True iff the lower bound of ci_95 is strictly greater than 0.

        This is a stronger criterion than signed_improvement > 0:
        - signed_improvement > 0 means the sample mean improved.
        - is_statistically_positive means even the pessimistic end of the
          95% confidence interval excludes zero improvement.

        Used to set honest_verdict='credible_positive' in the experiment artifact.

        Spec: REQ-BENCH-019, SCENARIO-BENCH-037
        """
        lower, _ = self.ci_95
        return lower > 0.0

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict including all computed properties."""
        lo, hi = self.ci_95
        return {
            "model_id": self.model_id,
            "pre_acc": self.pre_acc,
            "post_acc": self.post_acc,
            "n": self.n,
            "extractor_name": self.extractor_name,
            "inference_mode": self.inference_mode,
            "signed_improvement": self.signed_improvement,
            "ci_95": [lo, hi],
            "is_statistically_positive": self.is_statistically_positive,
            "cot_pairs_count": len(self.cot_pairs),
        }
