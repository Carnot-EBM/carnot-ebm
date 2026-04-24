"""VGSearchScheduler — Variable Granularity Search scheduling gate for ThreeTierPipeline.

**Researcher summary:**
    arXiv 2505.11730 (Variable Granularity Search) shows that optimal verification
    frequency depends on question difficulty: high-uncertainty questions need frequent
    checks; low-uncertainty questions can skip expensive verification.

    Applied to Carnot: track rolling energy variance (last N checks on the current
    response).  If variance < threshold → skip the current tier and fast-path to
    next.  If variance > threshold → run full tier check.

    This reduces Ising calls without accuracy degradation and complements MARS
    (which skips based on LLM logit margin rather than energy history).

**Why energy variance as the uncertainty signal:**
    The energy function is ground truth.  If the last N energy readings on a session
    are tightly clustered (low variance), the model is consistently confident about
    this response type — it is in a stable attractor.  Spending Ising compute to
    re-verify what is already well-characterised wastes resources.

    Variance is computed with numpy.var (population variance, not sample variance)
    because we always have exactly N readings when the window is full; the divisor
    is N, not N-1.

**Window semantics:**
    The scheduler maintains a sliding window of the last `window_size` energy
    readings.  Readings are accumulated from whichever tier ran last (Ising or EORM).
    When fewer than `window_size` readings exist, the scheduler always lets the
    tier run (insufficient history → cannot conclude low-variance).

Spec: REQ-VERIFY-171, REQ-VERIFY-172
SCENARIO-VERIFY-200
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# VGScheduleResult
# ---------------------------------------------------------------------------


@dataclass
class VGScheduleResult:
    """Decision record from a single VGSearchScheduler.should_skip() call.

    **Detailed explanation for engineers:**
        Encapsulates both the scheduling decision AND the raw signals used to
        make it, so callers can log the reasoning without re-computing it.

    Attributes
    ----------
    should_run_tier : bool
        True  → run the tier's full computation normally.
        False → skip this tier; reuse the last known energy value.
    energy_variance : float
        Population variance of the energy history window.  0.0 when there is
        insufficient history (fewer than window_size readings).
    variance_threshold : float
        The threshold that was compared against energy_variance.
    skip_reason : str
        One of:
            "insufficient_history" — window not yet full; tier MUST run.
            "low_variance_skip"    — window full, variance < threshold; tier skipped.
            "high_variance_run"    — window full, variance >= threshold; tier runs.
    honest_verdict : str
        Human-readable summary matching the conductor's artifact language.
    """

    should_run_tier: bool
    energy_variance: float
    variance_threshold: float
    skip_reason: str
    honest_verdict: str


# ---------------------------------------------------------------------------
# VGSearchScheduler
# ---------------------------------------------------------------------------


class VGSearchScheduler:
    """Variable Granularity Search scheduling gate.

    **Detailed explanation for engineers:**
        Implements the VGS scheduling heuristic from arXiv 2505.11730 adapted
        for energy-based verification:

        1. After each tier produces an energy value, call update(energy).
        2. Before running the next tier, call should_skip().
        3. If should_skip().should_run_tier is False, skip the tier and carry
           forward the previous energy value.

        The scheduler is stateful per-session (or per-question, depending on
        when reset() is called).  In ThreeTierPipeline wiring, reset() should
        be called at the start of each new question so history from one
        question does not influence the next.

    Parameters
    ----------
    variance_threshold : float
        Population variance threshold below which a tier is skipped.
        Default 0.05 — tuned so that energies within ±0.22 of their mean
        (sqrt(0.05) ≈ 0.22) are considered stable.
    window_size : int
        Number of recent energy readings to maintain.  Default 3 (from
        arXiv 2505.11730 ablation: N=3 gives best savings/accuracy tradeoff).

    Spec: REQ-VERIFY-171, REQ-VERIFY-172
    """

    def __init__(
        self,
        variance_threshold: float = 0.05,
        window_size: int = 3,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.window_size = window_size
        self._energy_history: list[float] = []

    # ------------------------------------------------------------------
    # update()
    # ------------------------------------------------------------------

    def update(self, energy: float) -> None:
        """Record a new energy reading into the rolling window.

        **Detailed explanation for engineers:**
            Maintains a FIFO sliding window of the last `window_size` readings.
            Oldest entries are dropped when the window is full.  Call this
            immediately after any tier produces an energy value.

        Parameters
        ----------
        energy : float
            The energy value produced by the tier that just ran.
        """
        self._energy_history.append(energy)
        if len(self._energy_history) > self.window_size:
            self._energy_history.pop(0)

    # ------------------------------------------------------------------
    # should_skip()
    # ------------------------------------------------------------------

    def should_skip(self) -> VGScheduleResult:
        """Decide whether the next tier should be skipped based on energy history.

        **Detailed explanation for engineers:**
            Three possible outcomes:

            1. Insufficient history (len < window_size):
               → should_run_tier=True, skip_reason="insufficient_history"
               We never skip when we have too few readings — can't know it's stable.

            2. Low variance (var < threshold):
               → should_run_tier=False, skip_reason="low_variance_skip"
               Energy is stable; skip expensive tier; reuse last energy.

            3. High variance (var >= threshold):
               → should_run_tier=True, skip_reason="high_variance_run"
               Energy is fluctuating; uncertain response; run full tier.

        Returns
        -------
        VGScheduleResult
            Complete decision record with raw signals for logging.

        Spec: REQ-VERIFY-171
        SCENARIO-VERIFY-200
        """
        if len(self._energy_history) < self.window_size:
            return VGScheduleResult(
                should_run_tier=True,
                energy_variance=0.0,
                variance_threshold=self.variance_threshold,
                skip_reason="insufficient_history",
                honest_verdict="insufficient_history_run_tier",
            )

        variance = float(np.var(self._energy_history))

        if variance < self.variance_threshold:
            return VGScheduleResult(
                should_run_tier=False,
                energy_variance=variance,
                variance_threshold=self.variance_threshold,
                skip_reason="low_variance_skip",
                honest_verdict="low_variance_skip_tier",
            )

        return VGScheduleResult(
            should_run_tier=True,
            energy_variance=variance,
            variance_threshold=self.variance_threshold,
            skip_reason="high_variance_run",
            honest_verdict="high_variance_run_tier",
        )

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear energy history.  Call at the start of each new question.

        **Detailed explanation for engineers:**
            Without reset(), history from one question bleeds into the next.
            If the previous question had stable low-variance energies and the
            current question is genuinely high-variance, the stale history
            would cause incorrect skipping.  Reset after each question.
        """
        self._energy_history = []
