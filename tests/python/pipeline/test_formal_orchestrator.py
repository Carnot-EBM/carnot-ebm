"""Tests for Formal Orchestrator."""

import z3
import pytest
from carnot.pipeline.formal_orchestrator import FormalOrchestrator, run_experiment_1787

def test_orchestrator_rejects_contradictory_proof():
    """
    Test that the formal orchestrator rejects contradictory/unsatisfiable proofs.
    References:
        - REQ-PIPELINE-1797
        - SCENARIO-PIPELINE-1797
    """
    orchestrator = FormalOrchestrator(max_iterations=3)

    x = z3.Int('x')

    def generator() -> z3.ExprRef:
        return x

    def validator(candidate: z3.ExprRef) -> z3.BoolRef:
        # A contradictory proof (x > 5 and x < 3)
        return z3.And(candidate > 5, candidate < 3)  # type: ignore[no-any-return]

    result = orchestrator.run_generation_loop(generator, validator)

    # It should not succeed if the proof is unsatisfiable
    assert result["success"] is False
    assert result["iterations"] == 3

def test_run_experiment_1787(tmp_path):
    """
    Test run_experiment_1787.
    References:
        - REQ-PIPELINE-1787
        - SCENARIO-PIPELINE-1787
    """
    output_path = tmp_path / "results" / "experiment_1787.json"
    result = run_experiment_1787(output_path=str(output_path))
    assert result["success"] is True
    assert output_path.exists()
