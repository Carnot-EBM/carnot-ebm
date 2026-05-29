"""Tests for RepairLadder.

Spec: REQ-VERIFY-3355, SCENARIO-VERIFY-3355
"""

from carnot.pipeline.repair_ladder import RepairLadder
from carnot.pipeline.verify_repair import VerifyRepairPipeline, VerificationResult
from carnot.pipeline.extract import ConstraintResult

def test_repair_ladder_success():
    """Test that RepairLadder can successfully repair a response."""
    pipeline = VerifyRepairPipeline(model=None, max_repairs=3)
    
    def mock_llm_caller(prompt: str) -> str:
        return "[Fixed by mock LLM]"
        
    ladder = RepairLadder(pipeline, max_iterations=2, llm_caller=mock_llm_caller)
    
    # Mock verify
    original_verify = pipeline.verify
    def simulated_verify(question: str, response: str, domain: str | None = None) -> VerificationResult:
        if "[Fixed by mock LLM]" in response:
            return VerificationResult(
                verified=True,
                constraints=[],
                energy=0.0,
                violations=[]
            )
        violation = ConstraintResult(
            constraint_type="test",
            description="Test constraint failure",
            metadata={"satisfied": False}
        )
        return VerificationResult(
            verified=False,
            constraints=[violation],
            energy=1.0,
            violations=[violation]
        )
        
    pipeline.verify = simulated_verify # type: ignore
    
    try:
        res = ladder.repair("Question", "Initial response", domain="math")
    finally:
        pipeline.verify = original_verify # type: ignore
        
    assert res.repaired is True
    assert res.iterations == 1
    assert res.final_response == "[Fixed by mock LLM]"
    assert res.satisfiable_drift > 0

def test_repair_ladder_failure():
    """Test that RepairLadder fails gracefully after max_iterations."""
    pipeline = VerifyRepairPipeline(model=None, max_repairs=3)
    
    def mock_llm_caller(prompt: str) -> str:
        return "[Still broken]"
        
    ladder = RepairLadder(pipeline, max_iterations=2, llm_caller=mock_llm_caller)
    
    # Mock verify
    original_verify = pipeline.verify
    def simulated_verify(question: str, response: str, domain: str | None = None) -> VerificationResult:
        violation = ConstraintResult(
            constraint_type="test",
            description="Test constraint failure",
            metadata={"satisfied": False}
        )
        return VerificationResult(
            verified=False,
            constraints=[violation],
            energy=1.0,
            violations=[violation]
        )
        
    pipeline.verify = simulated_verify # type: ignore
    
    try:
        res = ladder.repair("Question", "Initial response", domain="math")
    finally:
        pipeline.verify = original_verify # type: ignore
        
    assert res.repaired is False
    assert res.iterations == 1
    assert res.final_response == "[Still broken]"

def test_repair_ladder_ci_mode():
    """Test that RepairLadder breaks early if the LLM caller returns CI mode stub."""
    pipeline = VerifyRepairPipeline(model=None, max_repairs=3)
    
    ladder = RepairLadder(pipeline, max_iterations=2, llm_caller=None)
    
    original_verify = pipeline.verify
    def simulated_verify(question: str, response: str, domain: str | None = None) -> VerificationResult:
        violation = ConstraintResult(
            constraint_type="test",
            description="Test constraint failure",
            metadata={"satisfied": False}
        )
        return VerificationResult(
            verified=False,
            constraints=[violation],
            energy=1.0,
            violations=[violation]
        )
        
    pipeline.verify = simulated_verify # type: ignore
    
    try:
        res = ladder.repair("Question", "Initial response", domain="math")
    finally:
        pipeline.verify = original_verify # type: ignore
        
    assert res.repaired is False
    assert res.final_response == "Initial response"
    assert res.iterations == 0
