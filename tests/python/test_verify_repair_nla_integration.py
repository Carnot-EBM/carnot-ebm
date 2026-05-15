import pytest
from carnot.pipeline.verify_repair import VerifyRepairPipeline

def test_verify_repair_pipeline_has_16_verifiers():
    """Verify that VerifyRepairPipeline includes the NLA verifier as the 16th verifier.
    
    Spec: REQ-VERIFY-1720
    """
    pipeline = VerifyRepairPipeline(model=None)
    assert hasattr(pipeline, "verifier_list")
    assert len(pipeline.verifier_list) == 16
    assert "nla_verifier" in pipeline.verifier_list
    assert pipeline.verifier_list[-1] == "nla_verifier"
