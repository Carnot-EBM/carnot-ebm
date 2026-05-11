import json
import os
import time
from carnot.pipeline.roce import PromptConstraintExtractor

def test_roce_latency_and_validity_1847():
    """
    SCENARIO-ROCE-1847: Validate ROCE Latency limits on Dense Model
    """
    # REQ-ROCE-1847: Benchmark ROCE latency and validity on Dense
    extractor = PromptConstraintExtractor()
    prompt = "Your response must include the word 'summary' and maximum of 50 words"
    
    start_time = time.time()
    constraints = extractor.extract_from_prompt(prompt)
    
    # Check response validity
    valid_response = "Here is a short summary."
    invalid_response = "This response does not contain the required word and is very long. " * 10
    
    violated_valid = extractor.check_response(valid_response, constraints)
    violated_invalid = extractor.check_response(invalid_response, constraints)
    
    latency = time.time() - start_time
    
    assert len(violated_valid) == 0
    assert len(violated_invalid) > 0
    assert latency < 0.1  # Latency should be minimal
    
    # Ensure JSON deliverable exists and has correct fields
    json_path = "results/experiment_1847_gemma31_roce.json"
    assert os.path.exists(json_path)
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    assert data["experiment"] == 1847
    assert data["model"] == "unsloth/gemma-4-31B-it-GGUF"
    assert "latency_ms_per_token" in data
    assert "validity_score" in data
