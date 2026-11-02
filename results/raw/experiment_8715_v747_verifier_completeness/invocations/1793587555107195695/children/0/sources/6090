"""LagrangeAdaptiveIsing — per-constraint lambda weight adapter for the verify pipeline.

**Researcher summary:**
    This module provides a lightweight per-constraint Lagrange multiplier tracker
    that can be wired into SelfLearningRelay.  Each call to update() adjusts the
    lambda weight for a constraint based on whether the constraint was violated.
    After many sessions, constraints that are repeatedly violated receive higher
    lambda weights — making them stronger discriminators in the Ising energy landscape.

    This is the verify-layer counterpart to LagrangeAdaptiveIsingConstraints
    (samplers/lagrange_adaptive.py), which operates on the full coupling matrix.
    LagrangeAdaptiveIsing is stateless between batches except for lambda weights,
    making it suitable for relay experiments where we want to measure per-session
    improvement in isolation.

**Why this is separate from LagrangeAdaptiveIsingConstraints:**
    LagrangeAdaptiveIsingConstraints requires a full Ising spin configuration and
    InertiaIsingSampler.  LagrangeAdaptiveIsing works at the verify-pipeline level
    where we only know (constraint_id, violated) per question — not the full spin
    state.  This makes it CI-safe and usable in CPU-only relay experiments without
    any NumPy or GPU dependency in the hot path.

**Self-learning mechanism:**
    Lambda weights are the system's memory of which constraints are problematic.
    A rising mean_lambda() across sessions means the system is accumulating evidence
    that some constraints are systematically violated — and penalising them harder
    in future sessions.  This is the FR-11 Tier 1 signal: the system modifies its
    own energy landscape based on past failures without human supervision.

Spec: REQ-LEARN-058
"""

from __future__ import annotations


class LagrangeAdaptiveIsing:
    """Per-constraint Lagrange multiplier tracker for the verify pipeline relay.

    **Detailed explanation for engineers:**
        Each constraint is identified by an integer constraint_id.  When the pipeline
        reports a violation, update() increases the lambda for that constraint by
        lambda_lr.  When the constraint is satisfied, lambda decreases slightly
        (by 10% of lambda_lr), preventing runaway growth while preserving the
        asymmetry of Lagrange relaxation (violations matter more than successes).

        Lambda values are floored at lambda_init: they can grow above the initial
        value but not fall below it.  This models the fact that in Lagrange relaxation,
        the penalty for a constraint should never be weaker than the default level.

        Unknown constraint IDs are auto-initialized on first access with lambda_init.
        This means the caller does not need to know the total number of constraints
        upfront — n_constraints is a hint for pre-allocation only.

    Args:
        n_constraints: Number of constraints to pre-initialize.  Additional IDs
                       are initialized on first access.
        lambda_init:   Initial (and minimum) Lagrange multiplier.  Default 1.0.
        lambda_lr:     Learning rate for the lambda update.  Default 0.1.
                       Larger values respond faster to violations but may overshoot.

    Spec: REQ-LEARN-058
    """

    def __init__(
        self,
        n_constraints: int,
        lambda_init: float = 1.0,
        lambda_lr: float = 0.1,
    ) -> None:
        self.n_constraints = n_constraints
        self.lambda_init = lambda_init
        self.lambda_lr = lambda_lr
        # Per-constraint lambda weights — grow when violations accumulate.
        self._lambdas: dict[int, float] = {i: lambda_init for i in range(n_constraints)}
        # Counters for computing violation_rate() without storing full history.
        self._violation_counts: dict[int, int] = {i: 0 for i in range(n_constraints)}
        self._update_counts: dict[int, int] = {i: 0 for i in range(n_constraints)}

    def update(self, constraint_id: int, violated: bool) -> None:
        """Update lambda for constraint_id based on whether it was violated.

        **What happens per call:**
            - violated=True:  lambda increases by lambda_lr (Lagrange penalty grows).
            - violated=False: lambda decreases by 0.1 * lambda_lr, floored at lambda_init.

        **Why asymmetric updates:**
            Lagrange relaxation only penalises violations; there is no reward for
            satisfying a constraint beyond removing the penalty.  The small decrease
            when satisfied prevents lambdas from growing without bound when violations
            are occasional rather than systematic.  The 10x asymmetry (0.1 * lr for
            decrease vs 1x lr for increase) matches the ratio used in arXiv 2501.04971.

        Args:
            constraint_id: Integer identifier for the constraint.  Unknown IDs are
                           initialized with lambda_init on first access.
            violated:      True if the constraint was violated this step.

        Spec: REQ-LEARN-058
        """
        if constraint_id not in self._lambdas:
            self._lambdas[constraint_id] = self.lambda_init
            self._violation_counts[constraint_id] = 0
            self._update_counts[constraint_id] = 0

        self._update_counts[constraint_id] += 1
        if violated:
            self._violation_counts[constraint_id] += 1
            self._lambdas[constraint_id] += self.lambda_lr
        else:
            self._lambdas[constraint_id] = max(
                self.lambda_init,
                self._lambdas[constraint_id] - self.lambda_lr * 0.1,
            )

    def get_lambda(self, constraint_id: int) -> float:
        """Return the current lambda for constraint_id (lambda_init if never updated).

        Spec: REQ-LEARN-058
        """
        return self._lambdas.get(constraint_id, self.lambda_init)

    def violation_rate(self, constraint_id: int) -> float:
        """Return empirical violation rate for constraint_id across all update() calls.

        Returns 0.0 if constraint_id has never been updated.

        Spec: REQ-LEARN-058
        """
        total = self._update_counts.get(constraint_id, 0)
        if total == 0:
            return 0.0
        return self._violation_counts.get(constraint_id, 0) / total

    def mean_lambda(self) -> float:
        """Return the mean lambda across all known constraints.

        Rising mean_lambda across sessions indicates the system is accumulating
        Lagrange penalties for systematic violations — the FR-11 self-learning signal.

        Spec: REQ-LEARN-058
        """
        if not self._lambdas:
            return self.lambda_init
        return sum(self._lambdas.values()) / len(self._lambdas)


__all__ = ["LagrangeAdaptiveIsing"]
