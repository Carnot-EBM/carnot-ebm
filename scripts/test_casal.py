import json
import os
import sys

# Ensure results dir exists
os.makedirs("results", exist_ok=True)

from carnot.pipeline.verify_repair import VerifyRepairPipeline, CASALTier

def test_casal_tier():
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
    
    with open("results/experiment_1690_verify_repair_casal.json", "w") as f:
        json.dump(casal_res, f, indent=2)

if __name__ == "__main__":
    test_casal_tier()
    print("Test passed and artifact written.")
