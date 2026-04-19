"""Precision100qV5Result — per-model result type for the Exp 488 benchmark.

**Why a V5 result (Exp 488, RETRO-033 fifth attempt):**
    Exp 476 (V4) used `GPUVRAMGate` (V1) which checked VRAM before killing zombies.
    RETRO-044 confirmed this causes a race: the driver holds dead-process VRAM for
    5-15 s after SIGKILL, so a poll during that window deferred the experiment even
    though VRAM would have been free moments later.

    V5 adds `gpu_id : int` to the result so each model's GPU assignment is
    recorded in the artifact for auditability, and so tests can assert that Gemma4
    is on cuda:0 and Qwen is on cuda:1.

**Formula (Wilson score CI, same as V4):**
    Agresti & Coull (1998).  At n=100 the interval width is < 0.10, making
    directional claims (is_positive) statistically credible.

Spec: REQ-BENCH-034, REQ-BENCH-035, REQ-BENCH-036,
      SCENARIO-BENCH-053, SCENARIO-BENCH-054, SCENARIO-BENCH-055
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# z-score for 95% two-sided confidence interval
_Z95 = 1.959963984540054


@dataclass
class Precision100qV5Result:
    """Per-model result from the 100-question live precision benchmark (Exp 488).

    Extends V4 with an explicit `gpu_id` field that records which CUDA device the
    model was pinned to via DualGPUHarness.  This makes the device assignment
    auditable from the JSON artifact without reading the experiment log.

    Parameters
    ----------
    model_id : str
        Human-readable model name, e.g. 'Gemma4-E4B-it'.
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
        Records the DualGPUHarness assignment for auditability (REQ-BENCH-035).

    Computed properties
    -------------------
    signed_improvement : float
        post_accuracy - pre_accuracy (unclamped; negative means regression).
    ci_95_wilson : tuple[float, float]
        Wilson score 95% CI on post_accuracy.  Always within [0.0, 1.0].
        Width < 0.10 at n=100 (REQ-BENCH-034 traceability gate).
    is_positive : bool
        True iff signed_improvement > 0 (strict; zero is not positive).

    Spec: REQ-BENCH-034, REQ-BENCH-035, SCENARIO-BENCH-055
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

        Why we use Wilson rather than normal approximation:
            Normal approximation produces intervals outside [0,1] at extreme p.
            Wilson is always within the unit interval and better calibrated at
            small n.  n=100 with p near 0 or 1 is common in these benchmarks.

        Spec: REQ-BENCH-034, SCENARIO-BENCH-055
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

        The `ci_95_wilson` field is serialized as a two-element list (not tuple)
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
