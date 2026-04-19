"""Precision100qV6Result — per-model result for Exp 502 (Gemma4QuantizedLoader).

**Why V6 (Exp 502, RETRO-033 sixth attempt):**
    Exp 488 (V5) was deferred with gpu_vram_insufficient because the conductor process
    itself holds ~15.7 GiB GPU 0 VRAM, leaving only ~8.3 GiB free — not enough for
    Gemma4-E4B-it at FP16 (~14.89 GiB).  GPUVRAMGateV2 correctly killed zombie processes
    but could not free the conductor's own allocation (it is the parent process).

    V6 switches to Gemma4QuantizedLoader (GGUF Q4_K_M, ~8-10 GiB) so the model fits
    alongside the conductor with ~6 GiB headroom.  VRAMBudgetLedger pre-checks feasibility
    before the gate to give a fast-fail with an actionable root cause if the budget is still
    exceeded.

    The result type is identical to V5 (same fields and Wilson CI formula) because the
    benchmark protocol is unchanged — only the model loader changed.  V6 is kept as a
    separate class so the artifact schema version 'carnot.live_precision.v6' is distinct
    and traceable to Exp 502.

**Formula (Wilson score CI, same as V4/V5):**
    Agresti & Coull (1998).  At n=100 the interval width is < 0.10, making
    directional claims (is_positive) statistically credible.

Spec: REQ-BENCH-043, REQ-BENCH-044, REQ-BENCH-045,
      SCENARIO-BENCH-062, SCENARIO-BENCH-063, SCENARIO-BENCH-064
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# z-score for 95% two-sided confidence interval (Agresti & Coull, 1998)
_Z95 = 1.959963984540054


@dataclass
class Precision100qV6Result:
    """Per-model result from the 100-question live precision benchmark (Exp 502).

    Identical to V5 except the class name ties it to Exp 502 and the quantized
    loader.  Every field maps directly to a schema key in the artifact JSON.

    Parameters
    ----------
    model_id : str
        Human-readable model name, e.g. 'Gemma4-INT4' or 'Qwen3.5-0.8B'.
    pre_accuracy : float
        Fraction correct with NO pipeline applied (baseline). Range [0.0, 1.0].
    post_accuracy : float
        Fraction correct WITH pipeline applied. Range [0.0, 1.0].
    n : int
        Number of questions in the benchmark run. Must be > 0.
    extractor_used : str
        Comma-separated names of extractors that produced violations, or 'none'.
    inference_mode : str
        'live_gpu' when running on real hardware; 'synthetic' for fallback.
    gpu_id : int
        Zero-based CUDA device index the model was loaded on (0 = cuda:0).
        Records the DualGPUHarness assignment for auditability (REQ-BENCH-044).

    Computed properties
    -------------------
    signed_improvement : float
        post_accuracy - pre_accuracy (unclamped; negative means regression).
    ci_95_wilson : tuple[float, float]
        Wilson score 95% CI on post_accuracy.  Always within [0.0, 1.0].
        Width < 0.10 at n=100 (REQ-BENCH-043 traceability gate).
    is_positive : bool
        True iff signed_improvement > 0 (strict; zero is not positive).

    Spec: REQ-BENCH-043, REQ-BENCH-044, SCENARIO-BENCH-064
    """

    model_id: str
    pre_accuracy: float
    post_accuracy: float
    n: int
    extractor_used: str
    inference_mode: str
    gpu_id: int

    @property
    def signed_improvement(self) -> float:
        """post_accuracy minus pre_accuracy — negative means the pipeline regressed."""
        return self.post_accuracy - self.pre_accuracy

    @property
    def ci_95_wilson(self) -> tuple[float, float]:
        """Wilson score 95% confidence interval on post_accuracy.

        Formula (Agresti & Coull, 1998):
            center = (p̂ + z²/(2n)) / (1 + z²/n)
            margin = z * sqrt(p̂*(1-p̂)/n + z²/(4n²)) / (1 + z²/n)

        Returns (lower, upper) clamped to [0.0, 1.0].  At n=100 the width is
        typically < 0.10, making directional claims statistically credible.

        Why Wilson rather than normal approximation:
            Normal approximation produces intervals outside [0,1] at extreme p.
            Wilson is always within the unit interval and better calibrated when
            p is near 0 or 1 — which is common in GSM8K benchmarks.

        Spec: REQ-BENCH-043, SCENARIO-BENCH-064
        """
        p = self.post_accuracy
        n = max(self.n, 1)
        z = _Z95
        z2 = z * z

        center = (p + z2 / (2 * n)) / (1 + z2 / n)
        margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n)

        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        return (lower, upper)

    @property
    def is_positive(self) -> bool:
        """True iff signed_improvement > 0 (strict; zero improvement is not positive)."""
        return self.signed_improvement > 0

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict including all computed properties.

        The ci_95_wilson field is serialized as a two-element list (not tuple)
        so the artifact is valid JSON without a custom encoder.
        """
        lo, hi = self.ci_95_wilson
        return {
            "model_id": self.model_id,
            "pre_accuracy": self.pre_accuracy,
            "post_accuracy": self.post_accuracy,
            "n": self.n,
            "extractor_used": self.extractor_used,
            "inference_mode": self.inference_mode,
            "gpu_id": self.gpu_id,
            "signed_improvement": self.signed_improvement,
            "ci_95_wilson": [lo, hi],
            "is_positive": self.is_positive,
        }
