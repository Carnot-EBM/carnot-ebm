"""Tests for Formal Verification Orchestrator.

Spec traces: REQ-PIPELINE-1787, SCENARIO-PIPELINE-1787
"""

import json
import os
import z3
from typing import Any

from carnot.pipeline.formal_orchestrator import FormalOrchestrator, run_experiment_1787


def test_formal_orchestrator() -> None:
    """Test that FormalOrchestrator iteratively queries the solver.
    
    Spec: REQ-PIPELINE-1787, SCENARIO-PIPELINE-1787
    """
    x = z3.Int('x')
    
    def generator() -> z3.ExprRef:
        return x
    
    def validator(candidate: z3.ExprRef) -> z3.BoolRef:
        return candidate > 5  # type: ignore[no-any-return]

    orchestrator = FormalOrchestrator(max_iterations=3)
    result = orchestrator.run_generation_loop(generator, validator)
    
    assert result["success"] is True
    assert result["iterations"] == 1
    assert result["experiment_id"] == 1787


def test_run_experiment_1787(tmp_path: Any) -> None:
    """Test that the orchestrator experiment writes the artifact correctly.
    
    Spec: REQ-PIPELINE-1787, SCENARIO-PIPELINE-1787
    """
    output_path = tmp_path / "experiment_1787_formal_orchestrator.json"
    result = run_experiment_1787(str(output_path))
    
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        data = json.load(f)
    
    assert data["status"] == "complete"
    assert data["success"] is True
    assert data["experiment_id"] == 1787
