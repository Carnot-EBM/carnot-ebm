"""LagrangeAdaptiveUpdater — constraint weight accumulator with FOREVER-style forgetting curve.

**Researcher summary:**
    The standard Lagrange adaptive weight updater (Exp 866 / .66 milestone) accumulates
    constraint weights indefinitely.  In long sessions, early constraints from the first few
    questions dominate the energy landscape even when those questions are no longer contextually
    relevant.  The constraint memory grows without bound, increasing energy computation time
    and creating stale constraint interference that degrades precision.

    This module adds an exponential forgetting curve inspired by arXiv 2601.03938 (FOREVER:
    Forgetting curve learning for continual learning):

        w_t = w_0 * exp(-lambda * age_t)

    where age_t is the number of steps since the constraint was added.  Before a constraint
    is fully forgotten, any constraint that is STILL being violated at a high rate gets its
    weight reset (replayed) to prevent catastrophic forgetting of constraints that are
    currently important — FOREVER-style memory rehearsal.

    The lifecycle for each constraint:
        1. learn — added with default weight
        2. strengthen — updated based on violations
        3. gradually forget — exponential decay reduces weight over time
        4. replay if still relevant — high-violation constraints get weight reset
        5. eventually expire — when weight falls below 1e-4, removed from memory

**Why this matters for constraint memory management:**
    Without forgetting, a session processing 200 questions accumulates up to 200 constraints.
    Many of these encode error patterns from early questions that never recur.  These stale
    constraints add energy computation cost and may create phantom violations against later
    questions.  With forgetting (lambda=0.05, ~20 steps to half-life), stale constraints
    naturally expire while active violations are preserved by replay.

**The FOREVER connection:**
    arXiv 2601.03938 shows that combining exponential forgetting with rehearsal of
    high-importance samples prevents catastrophic forgetting in continual learning systems.
    We apply the same principle at the constraint level: the "samples" being rehearsed are
    constraints that still have high violation rates (still relevant) but are aging toward
    expiry.  Resetting their weight keeps them active in the energy landscape.

Spec: REQ-FR11-007, SCENARIO-FR11-007
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ConstraintRecord:
    """Per-constraint state tracked by LagrangeAdaptiveUpdater.

    **Detailed explanation for engineers:**
        Each constraint in the updater has three pieces of state:
        - weight:  current Lagrange multiplier (starts at w_0=1.0, modified by violations,
                   decayed by forgetting curve).
        - age:     how many tick() steps have elapsed since the constraint was added.
                   Age is incremented once per step, not once per violation.
        - violation_count / update_count: for computing the empirical violation rate,
          which the replay gate queries to decide if an aging constraint should be rescued.
    """

    weight: float = 1.0
    age: int = 0
    violation_count: int = 0
    update_count: int = 0

    @property
    def violation_rate(self) -> float:
        """Empirical fraction of update() calls where violated=True."""
        if self.update_count == 0:
            return 0.0
        return self.violation_count / self.update_count


class LagrangeAdaptiveUpdater:
    """Constraint weight accumulator with exponential forgetting and FOREVER-style replay.

    **Detailed explanation for engineers:**
        This class extends the basic Lagrange adaptive weight updater with a forgetting
        curve.  Constraints are identified by string IDs (e.g. "carry_error_q3") rather
        than integer indices because the pipeline adds constraints dynamically.

        ``update(constraint_id, violated)`` — same interface as LagrangeAdaptiveIsing.update():
            - violated=True:  weight increases by weight_lr (Lagrange penalty grows)
            - violated=False: weight decreases slightly (floor at weight_init)

        ``tick(step)`` — call once per step (e.g. after processing each question):
            - increments age for every constraint
            - applies exponential decay: w = w * exp(-forgetting_lambda)
            - removes constraints whose weight falls below EXPIRY_THRESHOLD (1e-4)

        ``get_replay_candidates(violation_rates)`` — call before tick() to identify
            constraints that are aging out but still relevant:
            - weight < REPLAY_WEIGHT_THRESHOLD (0.1): the constraint is aging toward expiry
            - violation_rate > replay_threshold (0.8): still being violated frequently
            - These constraints get their weight reset to weight_init (rescued from forgetting)

        ``constraint_precision`` — a read-only property measuring the fraction of currently
            active constraints that have a recent violation rate above a minimum threshold
            (i.e. they are "earning their keep" in the constraint memory).  A high
            constraint_precision means the memory is full of relevant constraints; a low
            value means many constraints are stale.

    Args:
        weight_init:         Initial weight for new constraints.  Default 1.0.
        weight_lr:           Learning rate for the weight update.  Default 0.1.
        forgetting_lambda:   Exponential decay rate per tick step.  Default 0.05.
                             At 0.05, a constraint reaches half its initial weight in
                             ~14 steps (ln(2)/0.05 ≈ 13.9).
        replay_threshold:    Violation rate above which an aging constraint is replayed
                             (weight reset to weight_init).  Default 0.8.
        precision_min_violation_rate:
                             Minimum violation rate for a constraint to count as
                             "relevant" in the constraint_precision calculation.  Default 0.1.

    Spec: REQ-FR11-007
    """

    EXPIRY_THRESHOLD: float = 1e-4
    REPLAY_WEIGHT_THRESHOLD: float = 0.1

    def __init__(
        self,
        weight_init: float = 1.0,
        weight_lr: float = 0.1,
        forgetting_lambda: float = 0.05,
        replay_threshold: float = 0.8,
        precision_min_violation_rate: float = 0.1,
    ) -> None:
        self.weight_init = weight_init
        self.weight_lr = weight_lr
        self.forgetting_lambda = forgetting_lambda
        self.replay_threshold = replay_threshold
        self.precision_min_violation_rate = precision_min_violation_rate

        # constraint_id -> ConstraintRecord
        self.constraint_weights: dict[str, float] = {}
        self.constraint_ages: dict[str, int] = {}
        self._records: dict[str, ConstraintRecord] = {}

        # Audit counters (never reset — allow full session analysis)
        self._total_replay_events: int = 0
        self._total_expired: int = 0

    # ------------------------------------------------------------------
    # Core update interface (Lagrange penalty)
    # ------------------------------------------------------------------

    def update(self, constraint_id: str, violated: bool) -> None:
        """Update weight for constraint_id based on whether it was violated.

        **What happens per call:**
            - violated=True:  weight increases by weight_lr (Lagrange penalty grows).
            - violated=False: weight decreases by 0.1 * weight_lr, floored at weight_init.

        This mirrors the asymmetry in LagrangeAdaptiveIsing: violations matter 10x more
        than satisfactions, matching the arXiv 2501.04971 ratio.

        Args:
            constraint_id: String identifier for the constraint.  Unknown IDs are
                           auto-initialized with weight_init on first access.
            violated:      True if the constraint was violated this step.

        Spec: REQ-FR11-007
        """
        if constraint_id not in self._records:
            rec = ConstraintRecord(weight=self.weight_init, age=0)
            self._records[constraint_id] = rec
            self.constraint_weights[constraint_id] = self.weight_init
            self.constraint_ages[constraint_id] = 0
        else:
            rec = self._records[constraint_id]

        rec.update_count += 1
        if violated:
            rec.violation_count += 1
            rec.weight += self.weight_lr
        else:
            rec.weight = max(self.weight_init, rec.weight - self.weight_lr * 0.1)

        # Keep the public dict in sync so callers can read constraint_weights directly.
        self.constraint_weights[constraint_id] = rec.weight

    # ------------------------------------------------------------------
    # Forgetting curve tick
    # ------------------------------------------------------------------

    def tick(self, step: int = 1) -> list[str]:
        """Increment age for all constraints, apply exponential decay, expire dead ones.

        Call this once per processing step (e.g. after each question is answered).

        **What happens:**
            1. For every known constraint, age += 1 and weight *= exp(-forgetting_lambda).
            2. If the new weight falls below EXPIRY_THRESHOLD (1e-4), the constraint is
               removed from both the _records dict and the public constraint_weights dict.
               This is the "forgetting" step — the constraint's energy contribution goes
               to effectively zero and we drop it to avoid unbounded memory growth.

        **Why step parameter?**
            Sometimes you want to advance multiple steps at once (e.g. simulating the
            effect of N questions without running all N update() calls).  Passing step>1
            applies exp(-forgetting_lambda * step) decay and increments ages by step.

        Args:
            step: Number of steps to advance.  Default 1.

        Returns:
            List of constraint IDs that were expired (removed) during this tick.

        Spec: REQ-FR11-007
        """
        decay_factor = math.exp(-self.forgetting_lambda * step)
        expired: list[str] = []

        for cid in list(self._records.keys()):
            rec = self._records[cid]
            rec.age += step
            rec.weight *= decay_factor
            self.constraint_ages[cid] = rec.age
            self.constraint_weights[cid] = rec.weight

            if rec.weight < self.EXPIRY_THRESHOLD:
                del self._records[cid]
                del self.constraint_weights[cid]
                del self.constraint_ages[cid]
                expired.append(cid)
                self._total_expired += 1

        return expired

    # ------------------------------------------------------------------
    # FOREVER-style replay gate
    # ------------------------------------------------------------------

    def get_replay_candidates(
        self,
        violation_rates: dict[str, float] | None = None,
    ) -> list[str]:
        """Return constraint IDs that are aging out but still have high violation rates.

        **What "aging out" means:**
            A constraint has weight < REPLAY_WEIGHT_THRESHOLD (0.1) — it has been
            decayed toward expiry by the forgetting curve and is no longer strongly
            penalising violations.  Normally it will expire within a few more ticks.

        **The FOREVER rescue rule:**
            If a constraint is aging out AND its empirical violation rate exceeds
            replay_threshold (default 0.8), it is still actively relevant — the system
            is still getting this constraint type wrong frequently.  We should NOT
            forget it yet.  Callers should call update(cid, True) or manually reset
            the weight to re-activate it.

        **Usage pattern in a relay loop:**
            candidates = updater.get_replay_candidates()
            for cid in candidates:
                updater._records[cid].weight = updater.weight_init   # rescue
                updater.constraint_weights[cid] = updater.weight_init
            updater.tick()  # now decayed constraints won't expire until next cycle

        Args:
            violation_rates: Optional override for per-constraint violation rates.
                             If None, uses the empirical rates from update() calls.
                             Useful when the caller has external violation data.

        Returns:
            List of constraint IDs that qualify for replay.

        Spec: REQ-FR11-007
        """
        candidates: list[str] = []
        for cid, rec in self._records.items():
            if rec.weight >= self.REPLAY_WEIGHT_THRESHOLD:
                # Not aging out — no replay needed.
                continue
            if violation_rates is not None:
                rate = violation_rates.get(cid, 0.0)
            else:
                rate = rec.violation_rate
            if rate > self.replay_threshold:
                candidates.append(cid)
        return candidates

    def apply_replay(
        self,
        violation_rates: dict[str, float] | None = None,
    ) -> int:
        """Find aging-but-active constraints and reset their weights (in-place rescue).

        This is a convenience wrapper that calls get_replay_candidates() and immediately
        resets the weight of each candidate to weight_init — the FOREVER rehearsal step.

        Returns:
            Number of constraints replayed (rescued from forgetting).

        Spec: REQ-FR11-007
        """
        candidates = self.get_replay_candidates(violation_rates)
        for cid in candidates:
            rec = self._records[cid]
            rec.weight = self.weight_init
            self.constraint_weights[cid] = self.weight_init
        self._total_replay_events += len(candidates)
        return len(candidates)

    # ------------------------------------------------------------------
    # Precision metric
    # ------------------------------------------------------------------

    @property
    def constraint_precision(self) -> float:
        """Fraction of active constraints with violation_rate >= precision_min_violation_rate.

        **Why this metric matters:**
            A high constraint_precision means most constraints in the active set are
            earning their keep — they capture error patterns that actually occur.
            A low value means many constraints are stale (they were added for questions
            that don't recur).

            With forgetting enabled, stale constraints expire naturally and precision
            rises over time.  Without forgetting, precision falls as the memory fills
            with stale constraints.

        Spec: REQ-FR11-007
        """
        if not self._records:
            return 0.0
        relevant = sum(
            1
            for rec in self._records.values()
            if rec.violation_rate >= self.precision_min_violation_rate
        )
        return relevant / len(self._records)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n_constraints(self) -> int:
        """Number of currently active (not yet expired) constraints."""
        return len(self._records)

    @property
    def total_replay_events(self) -> int:
        """Total number of FOREVER-style replay events since construction."""
        return self._total_replay_events

    @property
    def total_expired(self) -> int:
        """Total number of constraints that have been expired by the forgetting curve."""
        return self._total_expired

    @property
    def weight_entropy(self) -> float:
        """Shannon entropy of the current Lagrange weight distribution.

        **Why entropy matters here:**
            High entropy means weights are spread across many constraints — the
            penalty landscape is diverse and no single constraint dominates.  Low
            entropy (collapse toward 0) means a few constraints have accumulated
            almost all the weight, crowding out others; Tier 3 rejection rate rises
            because a small set of stale high-weight constraints makes the energy
            function insensitive to the actually-violated constraints.

            An exponential forgetting curve (non-zero forgetting_lambda) prevents
            weight collapse by decaying stale weights.  Tracking entropy over time
            shows whether the forgetting curve is doing its job.

        **Computation:**
            Treat normalized weights as a probability distribution and compute
            Shannon entropy H = -sum(p * log(p)).  The 1e-10 guard prevents
            log(0) when a weight is exactly zero.

        Returns:
            float: Shannon entropy in nats.  0.0 when there are no active
            constraints or all weight is on a single constraint.

        Spec: REQ-SELF-007
        """
        weights = np.array(list(self.constraint_weights.values()), dtype=float)
        if len(weights) == 0:
            return 0.0
        total = weights.sum()
        if total <= 0.0:
            return 0.0
        p = weights / total
        return float(-np.sum(p * np.log(p + 1e-10)))

    def summary(self) -> dict[str, Any]:
        """Return a serialisable summary of the updater state for artifact writing."""
        return {
            "n_constraints": self.n_constraints,
            "total_replay_events": self.total_replay_events,
            "total_expired": self.total_expired,
            "constraint_precision": self.constraint_precision,
            "forgetting_lambda": self.forgetting_lambda,
            "replay_threshold": self.replay_threshold,
        }


__all__ = ["LagrangeAdaptiveUpdater", "ConstraintRecord"]
