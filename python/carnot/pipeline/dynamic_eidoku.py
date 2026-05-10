"""Dynamic compiler for Eidoku gates based on extracted constraints.

**Researcher summary:**
    Synthesizes Eidoku gates dynamically from extracted constraints (ROCE).
    Instead of static rules, the gate's constraint graph is compiled at runtime
    from the constraints extracted from the user's prompt.

Spec: REQ-DYNAMIC-EIDOKU-001, SCENARIO-DYNAMIC-EIDOKU-001
"""

from __future__ import annotations

import time
from typing import Sequence

from carnot.pipeline.constraint_extractor import DynamicConstraint
from carnot.pipeline.eidoku_gate import EidokuGateResult


class CompiledEidokuGate:
    """An executable Eidoku gate compiled from a dynamic constraint graph.

    Spec: REQ-DYNAMIC-EIDOKU-001-3
    """

    def __init__(self, constraints: Sequence[DynamicConstraint], penalty_per_violation: float = 10.0) -> None:
        self.constraints = list(constraints)
        self.penalty_per_violation = penalty_per_violation

    def compute_cost(self, question: str, response: str) -> EidokuGateResult:
        """Evaluate the constraint graph against the response.

        Args:
            question: Original question (unused here, kept for interface compatibility).
            response: The response to evaluate against constraints.

        Returns:
            EidokuGateResult with the structural violation cost and runtime.

        Spec: REQ-DYNAMIC-EIDOKU-001-3
        """
        start = time.monotonic()
        cost = 0.0

        for constraint in self.constraints:
            if not constraint.check(response):
                cost += self.penalty_per_violation

        runtime_ms = (time.monotonic() - start) * 1000.0
        return EidokuGateResult(violation_cost=cost, runtime_ms=runtime_ms)


class DynamicEidokuCompiler:
    """Compiles extracted constraints into an executable Eidoku gate graph.

    Spec: REQ-DYNAMIC-EIDOKU-001-1
    """

    def __init__(self, penalty_per_violation: float = 10.0) -> None:
        self.penalty_per_violation = penalty_per_violation

    def compile(self, constraints: Sequence[DynamicConstraint]) -> CompiledEidokuGate:
        """Convert extracted constraints into an executable Eidoku gate.

        Args:
            constraints: List of DynamicConstraint objects forming the graph.

        Returns:
            A CompiledEidokuGate that can be executed to compute costs.

        Spec: REQ-DYNAMIC-EIDOKU-001-2, SCENARIO-DYNAMIC-EIDOKU-001
        """
        return CompiledEidokuGate(constraints, penalty_per_violation=self.penalty_per_violation)
