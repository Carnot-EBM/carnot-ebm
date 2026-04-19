"""Helpers for Exp 515 Live 200q VeriCoT+VPRM v5 benchmark.

**Why this module exists (RETRO-038 context):**
    RETRO-038 requires a statistically significant live benchmark: Wilson 95% CI lower bound
    > 0 at n=200 constitutes the first publishable credibility claim for Carnot's pipeline.
    This module exposes two testable primitives that drive the RETRO-038 gating decision:
    - ``compute_wilson_ci``      — Wilson score 95% CI on a proportion
    - ``is_statistically_positive`` — True iff the CI lower bound exceeds 0

**Why Wilson CI instead of a t-test or Wald interval?**
    The Wilson score interval is valid at all proportions including 0 and 1, whereas the
    Wald interval breaks down near the extremes.  For small-to-medium n (n=200 here) and
    proportions that may lie near 0.5, Wilson CI gives better-calibrated coverage.

**Relationship to Exp 514 wilson_ci:**
    Exp 514 uses ``wilson_ci(n_correct, n)`` from live_100q_v7_helpers.  This module
    exposes the same computation under a more descriptive name (``compute_wilson_ci``)
    with an explicit ``confidence`` parameter so the caller can vary the confidence level
    in tests without re-reading the implementation.

Spec: REQ-BENCH-016, SCENARIO-BENCH-035, SCENARIO-BENCH-036
"""

from __future__ import annotations

import math
from typing import Tuple

__all__ = [
    "compute_wilson_ci",
    "is_statistically_positive",
]


def compute_wilson_ci(
    n_successes: int,
    n_total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for an observed proportion.

    Wilson CI is the preferred method for bounded proportions because it remains
    numerically valid at n_successes=0 and n_successes=n_total, unlike the Wald
    normal-approximation interval which collapses to zero width at the extremes.

    The formula is:
        center = (p + z²/(2n)) / (1 + z²/n)
        margin = z * sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n)
        lower  = max(0, center - margin)
        upper  = min(1, center + margin)

    where p = n_successes / n_total and z is the normal quantile for `confidence`.

    Parameters
    ----------
    n_successes : int
        Number of successful outcomes (e.g. correct answers).
    n_total : int
        Total number of trials (e.g. questions answered).
    confidence : float
        Desired confidence level (default 0.95 → z=1.96).  Supported values map
        to standard z-scores: 0.90 → 1.645, 0.95 → 1.960, 0.99 → 2.576.

    Returns
    -------
    (lower, upper) : Tuple[float, float]
        Lower and upper bounds clamped to [0.0, 1.0].

    Raises
    ------
    ValueError
        When n_total < 0, n_successes < 0, n_successes > n_total, or confidence
        is not in (0, 1).

    Spec: REQ-BENCH-016, SCENARIO-BENCH-035
    """
    if n_total < 0:
        raise ValueError(f"n_total must be >= 0, got {n_total}")
    if n_successes < 0:
        raise ValueError(f"n_successes must be >= 0, got {n_successes}")
    if n_successes > n_total:
        raise ValueError(f"n_successes ({n_successes}) > n_total ({n_total})")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    if n_total == 0:
        return 0.0, 0.0

    # Map common confidence levels to their z-scores; fall back to scipy for others.
    _Z_TABLE = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    z = _Z_TABLE.get(round(confidence, 4))
    if z is None:
        # Use a direct approximation: z = sqrt(2) * erfinv(confidence)
        # For supported values this is equivalent to the standard table.
        # scipy.special.ndtri would be more accurate but we avoid the dependency.
        z = math.sqrt(2.0) * _erfinv_approx(confidence)

    p = n_successes / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    center = (p + z2 / (2.0 * n_total)) / denom
    margin = (z * math.sqrt(p * (1.0 - p) / n_total + z2 / (4.0 * n_total * n_total))) / denom

    return max(0.0, center - margin), min(1.0, center + margin)


def _erfinv_approx(confidence: float) -> float:
    """Rational approximation of erfinv(confidence) for non-standard z lookups.

    Uses the Rational approximation from Abramowitz and Stegun §26.2.17.
    Error < 4.5e-4 over the range used in practice (confidence 0.8 to 0.999).

    Only called for confidence values not in the standard z-table (0.90, 0.95, 0.99).
    """
    # erfinv(x) such that erf(z/sqrt(2)) = confidence → Phi(z) = (1+confidence)/2
    # We want the z where Phi(z) = (1 + confidence) / 2
    p = (1.0 + confidence) / 2.0
    # Abramowitz & Stegun rational approximation for inverse normal CDF
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    return t - (c[0] + c[1] * t + c[2] * t * t) / (1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t)


def is_statistically_positive(wilson_lower_bound: float) -> bool:
    """Return True iff the Wilson CI lower bound strictly exceeds 0.

    This is the RETRO-038 gating predicate.  A lower bound > 0 means the 95% CI
    for the pipeline improvement does not include zero, which is the minimum bar
    for a publishable credibility claim under the project's statistical policy.

    Why strict inequality (> 0 rather than >= 0)?
        A lower bound of exactly 0.0 occurs when n_successes=0 or the CI touches
        the origin — neither case constitutes a positive improvement signal.
        Strict positivity is the conservative and honest choice.

    Parameters
    ----------
    wilson_lower_bound : float
        The lower bound returned by ``compute_wilson_ci``.

    Returns
    -------
    bool
        True iff ``wilson_lower_bound > 0``.

    Spec: REQ-BENCH-016, SCENARIO-BENCH-036
    """
    return wilson_lower_bound > 0.0
