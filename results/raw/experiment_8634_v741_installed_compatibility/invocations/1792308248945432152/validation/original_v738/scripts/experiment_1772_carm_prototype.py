"""
Experiment 1772: CARM Prototype Evaluation.

Spec: REQ-CARM-1772-2
"""
import json
from pathlib import Path
from carnot.carm.prototype import CARMExtractor

def run_experiment(output_path="results/experiment_1772_care_prototype.json"):
    """Evaluate extraction accuracy of CARM prototype on CARE test suite."""
    test_suite_path = Path("results/experiment_1771_care_test_suite.json")
    if not test_suite_path.exists():
        raise FileNotFoundError(f"Test suite not found at {test_suite_path}")
        
    test_suite = json.loads(test_suite_path.read_text())
    
    extractor = CARMExtractor(model_spec="unsloth/Qwen3.6-35B-A3B-GGUF")
    
    correct = 0
    total = len(test_suite["cases"])
    
    for case in test_suite["cases"]:
        extracted = extractor.extract_constraints(case["instruction"])
        
        # Simple evaluation: if the extracted output exactly matches the ground truth
        if extracted == case["ground_truth"]:
            correct += 1
        else:
            print(f"Failed case {case['id']}: expected {case['ground_truth']}, got {extracted}")
            
    accuracy = correct / total if total > 0 else 0.0
    
    deliverable = {
        "schema": "carnot.carm.prototype.v1",
        "experiment_id": 1772,
        "model_specs": [extractor.model_spec],
        "extraction_accuracy": float(accuracy),
        "status": "complete",
        "honest_verdict": "complete: CARM prototype evaluated",
    }
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deliverable, indent=2))
    
    print(f"Extraction accuracy: {accuracy}")
    return deliverable

if __name__ == "__main__":
    run_experiment()
