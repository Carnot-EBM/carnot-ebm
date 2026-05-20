import pytest
from carnot.pipeline.verify_repair import VerifyRepairPipeline

def test_odar_route():
    pipeline = VerifyRepairPipeline(model=None)
    
    # Test fast_path (complexity is low, confidence is high)
    # F = 0.02 - 0.9 = -0.88 < 0.3
    assert pipeline.odar_route("Short prompt", context_energy=0.1) == "fast_path"
    
    # Test deliberative_path (complexity is high, confidence is low)
    # 50 words -> complexity 0.5. energy 0.9 -> confidence 0.1
    # F = 0.5 - 0.1 = 0.4 >= 0.3
    long_prompt = " ".join(["word"] * 50)
    assert pipeline.odar_route(long_prompt, context_energy=0.9) == "deliberative_path"
    
    # Test None context_energy (confidence 0.5)
    # 10 words -> complexity 0.1. F = 0.1 - 0.5 = -0.4 < 0.3
    assert pipeline.odar_route("Just a ten word prompt to test None context energy.") == "fast_path"
    
    # Test None context_energy deliberative
    # 90 words -> complexity 0.9. F = 0.9 - 0.5 = 0.4 >= 0.3
    very_long_prompt = " ".join(["word"] * 90)
    assert pipeline.odar_route(very_long_prompt) == "deliberative_path"
