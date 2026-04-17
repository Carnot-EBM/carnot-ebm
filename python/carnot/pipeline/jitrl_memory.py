"""JitRLConstraintMemory — per-domain threshold modulation via Just-in-Time RL.

**Researcher summary (Exp 415 → Exp 432):**
    JitRL (Just-in-Time Reinforcement Learning) adapts per-domain verification
    thresholds based on observed false-positive rates.  The key insight from
    arXiv 2601.18510 is that different problem types trigger different FP
    patterns: rate/ratio problems tend to produce high-energy false alarms
    (so we raise the threshold), while pure arithmetic problems produce
    low-energy violations that are almost always real (so we lower it).

    Exp 415 validated on synthetic data: threshold modulation works in both
    directions on simulated GSM8K data.
    Exp 432 validates on real data from Exp 427: fp_reduction_pct > 0 when
    feeding actual model violations through the JitRL loop.

**What this module provides:**
    ``JitRLConstraintMemory`` — accumulates per-domain violation history and
    exposes an adapted threshold.  The threshold starts at ``base_threshold``
    and is nudged up/down by ``lr`` after each recorded outcome.

    ``record(domain, violation_energy, was_fp)`` — updates history.
    ``threshold(domain)`` — returns current adapted threshold for domain.

    The adaptation rule is:
        threshold += lr  if was_fp  (raise threshold to reduce FP rate)
        threshold -= lr  if not was_fp AND violation_energy > threshold
                                    (lower threshold to catch real violations)

    This is a lightweight online rule, not full RL — but it matches the
    arXiv 2601.18510 "reactive threshold" framing well enough for Tier 1.

**Honest constraints:**
    - No cross-domain knowledge transfer.  Each domain adapts independently.
    - lr is fixed; no schedule.  Good enough for Tier 1 validation.
    - Thread-unsafe by design — single-process experiments only.

Spec: REQ-LEARN-034,
      SCENARIO-LEARN-060, SCENARIO-LEARN-061 (Exp 432)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# ViolationRecord
# ---------------------------------------------------------------------------


@dataclass
class ViolationRecord:
    """One recorded violation outcome for a domain.

    Attributes:
        domain:           Problem domain (e.g. 'rate_problems', 'arithmetic').
        violation_energy: Raw energy value produced by the verifier.
        was_fp:           True if the violation was confirmed a false positive.
    """

    domain: str
    violation_energy: float
    was_fp: bool


# ---------------------------------------------------------------------------
# JitRLConstraintMemory
# ---------------------------------------------------------------------------


class JitRLConstraintMemory:
    """Online per-domain threshold adaptation via Just-in-Time RL.

    **Detailed explanation for engineers:**
        Thresholds start at ``base_threshold`` (default 0.5, matching the
        VerifyRepairPipeline's default violation threshold).

        After each ``record()`` call:
          - If the violation was a false positive (was_fp=True), we RAISE the
            threshold for that domain by ``lr``.  Rationale: if the verifier is
            too sensitive for this domain, we need a higher bar before flagging.
          - If the violation was a true positive (was_fp=False) and the energy
            exceeded the current threshold, we LOWER the threshold by ``lr``.
            Rationale: the verifier is catching real errors; we can be more
            aggressive.

        Thresholds are clamped to [0.05, 0.95] to prevent degenerate extremes.

        ``history`` stores all raw records for post-hoc audit and artifact export.

    Args:
        base_threshold: Starting threshold for all domains (default 0.5).
        lr:             Learning rate — per-record threshold nudge (default 0.02).

    Spec: REQ-LEARN-034
    """

    def __init__(
        self,
        base_threshold: float = 0.5,
        lr: float = 0.02,
    ) -> None:
        self._base_threshold = base_threshold
        self._lr = lr
        # Per-domain adapted thresholds (initialised on first access)
        self._thresholds: Dict[str, float] = {}
        # Full violation history for audit
        self.history: List[ViolationRecord] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record(
        self,
        domain: str,
        violation_energy: float,
        *,
        was_fp: bool,
    ) -> None:
        """Record one violation outcome and update the domain threshold.

        **Detailed explanation for engineers:**
            Called once per verification decision where a constraint fired.
            The ``was_fp`` label must come from an external oracle (human
            annotation, ground-truth GSM8K answer, or programmatic check).

        Args:
            domain:           Problem domain string.
            violation_energy: Energy value from the verifier (≥ 0).
            was_fp:           True if this was a false positive.

        Spec: REQ-LEARN-034, SCENARIO-LEARN-060
        """
        self.history.append(
            ViolationRecord(domain=domain, violation_energy=violation_energy, was_fp=was_fp)
        )
        current = self._thresholds.get(domain, self._base_threshold)
        if was_fp:
            current = min(0.95, current + self._lr)
        elif violation_energy > current:
            current = max(0.05, current - self._lr)
        self._thresholds[domain] = current

    def threshold(self, domain: str) -> float:
        """Return the current adapted threshold for *domain*.

        If no violations have been recorded for this domain, returns the base
        threshold.

        Spec: REQ-LEARN-034, SCENARIO-LEARN-061
        """
        return self._thresholds.get(domain, self._base_threshold)

    def to_dict(self) -> dict:
        """Serialize state for JSON artifact embedding.

        Returns a dict with ``base_threshold``, ``lr``, ``thresholds``
        (per-domain), and ``n_records`` count.
        """
        return {
            "base_threshold": self._base_threshold,
            "lr": self._lr,
            "thresholds": dict(self._thresholds),
            "n_records": len(self.history),
        }
