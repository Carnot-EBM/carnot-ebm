"""Tests for the dynamic Eidoku compiler.

Spec: REQ-DYNAMIC-EIDOKU-001, SCENARIO-DYNAMIC-EIDOKU-001
"""

import pytest

from carnot.pipeline.constraint_extractor import DynamicConstraint
from carnot.pipeline.dynamic_eidoku import CompiledEidokuGate, DynamicEidokuCompiler


def test_compiler_initialization() -> None:
    """Test that the compiler initializes correctly.

    Spec: REQ-DYNAMIC-EIDOKU-001
    """
    compiler = DynamicEidokuCompiler(penalty_per_violation=15.0)
    assert compiler.penalty_per_violation == 15.0


def test_graph_compilation() -> None:
    """Test compiling extracted constraints into an executable gate.

    Spec: REQ-DYNAMIC-EIDOKU-001, SCENARIO-DYNAMIC-EIDOKU-001
    """
    compiler = DynamicEidokuCompiler(penalty_per_violation=10.0)
    constraints = [
        DynamicConstraint(
            instruction_type="must_contain",
            description="must contain X",
            metadata={"term": "X"},
            raw_phrase="must contain X",
        )
    ]
    gate = compiler.compile(constraints)

    assert isinstance(gate, CompiledEidokuGate)
    assert len(gate.constraints) == 1
    assert gate.penalty_per_violation == 10.0


def test_compiled_gate_execution_no_violations() -> None:
    """Test evaluating a response that satisfies all constraints in the compiled graph.

    Spec: REQ-DYNAMIC-EIDOKU-001
    """
    compiler = DynamicEidokuCompiler(penalty_per_violation=10.0)
    constraints = [
        DynamicConstraint(
            instruction_type="must_contain",
            description="must contain hello",
            metadata={"term": "hello"},
            raw_phrase="must contain hello",
        )
    ]
    gate = compiler.compile(constraints)

    result = gate.compute_cost(question="Say hello", response="Hello world!")
    assert result.violation_cost == 0.0
    assert result.runtime_ms >= 0.0


def test_compiled_gate_execution_with_violations() -> None:
    """Test evaluating a response that violates constraints in the compiled graph.

    Spec: REQ-DYNAMIC-EIDOKU-001
    """
    compiler = DynamicEidokuCompiler(penalty_per_violation=10.0)
    constraints = [
        DynamicConstraint(
            instruction_type="must_contain",
            description="must contain hello",
            metadata={"term": "hello"},
            raw_phrase="must contain hello",
        ),
        DynamicConstraint(
            instruction_type="max_words",
            description="at most 2 words",
            metadata={"limit": 2},
            raw_phrase="max 2 words",
        ),
    ]
    gate = compiler.compile(constraints)

    # Violates max_words constraint (4 words > 2)
    result = gate.compute_cost(question="Say hello", response="Hello beautiful new world!")
    assert result.violation_cost == 10.0

    # Violates both constraints (no hello, 4 words > 2)
    result = gate.compute_cost(question="Say goodbye", response="Goodbye beautiful new world!")
    assert result.violation_cost == 20.0
