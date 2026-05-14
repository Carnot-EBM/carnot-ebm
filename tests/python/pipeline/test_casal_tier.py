import pytest
from carnot.pipeline.verify_repair import VerifyRepairPipeline, CASALTier

def test_casal_tier_integration():
    """Verify CASALTier integration.
    
    Spec: REQ-CASAL-001, SCENARIO-CASAL-002
    """
    casal = CASALTier()
    pipeline = VerifyRepairPipeline(casal_tier=casal)
    
    question = "Mock question"
    response = "Mock response"
    
    result = pipeline.verify(question, response, domain="mock")
    
    casal_res = result.certificate.get("casal_tier")
    assert casal_res is not None
    assert casal_res["schema"] == "CASAL"
    assert casal_res["integration_successful"] is True
    assert "latency_ms" in casal_res
    assert casal_res["acceptance_gate_passed"] is True
