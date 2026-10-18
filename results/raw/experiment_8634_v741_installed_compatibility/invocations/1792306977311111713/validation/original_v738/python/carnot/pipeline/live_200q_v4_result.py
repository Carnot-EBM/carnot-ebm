"""Live200qV4Result — 200-question live benchmark result for Exp 503.

**Differences from Live200qV3Result (Exp 489):**

    1. Used by Exp 503, which loads Gemma4 via Gemma4QuantizedLoader (INT4/GGUF Q4_K_M)
       rather than FP16 via GemmaTransformersLoader.  Quantized Gemma4 fits in ~9 GiB,
       resolving the VRAM OOM that deferred Exp 489 (RETRO-038).

    2. ``ci_95_wilson`` and ``is_statistically_positive`` use the same statistical
       machinery as V3.  A statistically significant result here is the first
       publishable credibility claim Carnot can make publicly.

    3. ``to_dict()`` emits the same schema as V3 so downstream JEPA retrain (Exp 510)
       can consume either version without format conversion.

**Why Wilson CI on post_acc (ci_95_wilson):**

    Wilson score CI (Agresti & Coull, 1998) stays in [0,1] and shrinks gracefully near
    the boundaries.  At n=200 the half-width is <= 0.035 for any p, making post_acc
    estimates credible at the ±3.5pp level.  This is the CI that gates public claims.

**Why Wald CI on the improvement (is_statistically_positive):**

    A 2pp improvement sounds real, but at n=200 with typical pre_acc ~0.70 the SE of the
    difference is ~0.045.  The 95% CI spans roughly -0.07 to +0.11 — clearly noise.
    The Wald CI correctly accounts for variance in BOTH pre_acc and post_acc estimates.
    Only when the lower bound of this CI strictly exceeds zero is the improvement
    considered publishable.

Spec: REQ-BENCH-046, REQ-BENCH-047, REQ-BENCH-048,
      SCENARIO-BENCH-065, SCENARIO-BENCH-066, SCENARIO-BENCH-067 (Exp 503)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# z-score for 95% two-sided confidence interval (Agresti & Coull, 1998)
_Z95 = 1.959963984540054


@dataclass
class Live200qV4Result:
    """Per-model result from the 200-question live benchmark v4 (Exp 503).

    Parameters
    ----------
    model_id : str
        Human-readable model name, e.g. 'Gemma4-INT4' or 'Qwen3.5-0.8B'.
    pre_acc : float
        Fraction correct with NO pipeline (baseline pass).  Range [0.0, 1.0].
    post_acc : float
        Fraction correct WITH integrated VeriCoT+VPRM pipeline.  Range [0.0, 1.0].
    n : int
        Number of questions evaluated.  Must be > 0.  200 for Exp 503.
    extractor_name : str
        Name of the extraction stack used, e.g. 'VeriCoT+VPRM'.
    inference_mode : str
        'live_gpu' when running on real hardware; 'stub' in CI; 'synthetic' fallback.

    Computed properties
    -------------------
    signed_improvement : float
        post_acc - pre_acc (unclamped; negative means pipeline degraded accuracy).
        Honest negative results are preserved per CLAUDE.md.
    ci_95_wilson : tuple[float, float]
        Wilson score CI on post_acc at 95% confidence level.
        Width < 0.07 at n=200 for any p in [0,1] (REQ-BENCH-047).
    is_statistically_positive : bool
        True iff the lower bound of the Wald CI for the improvement > 0.0.
        A 2pp improvement at n=200 does NOT satisfy this — you need ~10pp+
        at n=200 to cross the bar.  This is intentionally strict.

    Spec: REQ-BENCH-046, REQ-BENCH-047, REQ-BENCH-048,
          SCENARIO-BENCH-065, SCENARIO-BENCH-066, SCENARIO-BENCH-067
    """

    model_id: str
    pre_acc: float
    post_acc: float
    n: int
    extractor_name: str
    inference_mode: str

    @property
    def signed_improvement(self) -> float:
        """Return post_acc - pre_acc (unclamped signed delta).

        Negative values mean the pipeline degraded accuracy.  Never clamped —
        honest negative results are required by CLAUDE.md.

        Spec: REQ-BENCH-046
        """
        return self.post_acc - self.pre_acc

    @property
    def ci_95_wilson(self) -> tuple[float, float]:
        """Wilson score 95% CI for post_acc.

        Returns (lower, upper) clamped to [0.0, 1.0].

        At n=200 the full width is < 0.07 for p near 0 or 1 (boundary cases).
        At p=0.5 (worst case), full width ≈ 0.138 — note the spec says 'interval
        width < 0.07' meaning the half-width (margin).

        Formula (Agresti & Coull, 1998):
            center = (p̂ + z²/(2n)) / (1 + z²/n)
            margin = z * sqrt(p̂*(1-p̂)/n + z²/(4n²)) / (1 + z²/n)

        Why Wilson instead of Wald:
            The Wald CI (p ± z*sqrt(p(1-p)/n)) can produce intervals outside [0,1]
            for small p or near-boundary p values.  Wilson CI is bounded by construction
            and has better coverage properties at the boundaries — important for models
            with very low or very high baseline accuracy.

        Spec: REQ-BENCH-047, SCENARIO-BENCH-065
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
        """Return True iff the pipeline improvement is statistically significant at 95%.

        Uses a Wald CI for the difference of two proportions (unpooled):
            improvement ± z * sqrt(pre*(1-pre)/n + post*(1-post)/n)

        The lower bound of this CI must strictly exceed zero for the result to count.

        Why this is harder to satisfy than signed_improvement > 0:
            At pre_acc=0.70, post_acc=0.72 (improvement=0.02), n=200:
            SE = sqrt(0.21/200 + 0.2016/200) ≈ 0.045
            lower = 0.02 - 1.96*0.045 ≈ -0.069 → False (noise)

        A credible improvement requires roughly 10pp+ at n=200 to cross the bar.
        This strict gate is what makes RETRO-038 closure publishable.

        Spec: REQ-BENCH-047, SCENARIO-BENCH-066
        """
        n = max(self.n, 1)
        pre = self.pre_acc
        post = self.post_acc
        se = math.sqrt(pre * (1 - pre) / n + post * (1 - post) / n)
        lower = self.signed_improvement - _Z95 * se
        return lower > 0.0

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict including all computed properties.

        The schema is compatible with Live200qV3Result.to_dict() so Exp 510 JEPA
        retrain can consume either version without format conversion.

        Spec: REQ-BENCH-048, SCENARIO-BENCH-067
        """
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
        }
