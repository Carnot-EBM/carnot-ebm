#!/usr/bin/env python3
"""
Experiment 1765: Evaluate ROCE and HRM Components Together.

Spec: REQ-ROCE-HRM-1765
"""

import json
import os
import sys

# Ensure carnot is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

from carnot.pipeline.roce_extractor import ROCEExtractor
from carnot.models.hrm_verifier import HRMVerifier

def run_experiment(output_path: str, model_name: str):
    """Run the ROCE and HRM evaluation experiment.
    
    Spec: SCENARIO-ROCE-HRM-1765
    """
    extractor = ROCEExtractor()
    verifier = HRMVerifier(levels=3)
    
    # Sample prompts simulating the evaluation dataset
    prompts = [
        "Please provide a JSON response. The response must contain the word 'apple'.",
        "Give exactly 3 bullet points. Do not include the word 'banana'."
    ]
    
    total_constraints = 0
    total_score = 0.0
    
    results_details = []
    
    for prompt in prompts:
        constraints = extractor.extract(prompt, domain="roce")
        constraints_list = [c.metadata for c in constraints]
        eval_result = verifier.evaluate(constraints_list)
        
        total_constraints += len(constraints)
        total_score += eval_result["score"]
        
        results_details.append({
            "prompt": prompt,
            "constraints_extracted": len(constraints),
            "score": eval_result["score"],
            "hrm_details": eval_result["details"]
        })
        
    avg_score = total_score / len(prompts) if prompts else 0.0
    
    report = {
        "experiment_id": "1765",
        "model": model_name,
        "constraint_satisfaction_rate": avg_score, # Metric based on verifier score
        "total_prompts_evaluated": len(prompts),
        "total_constraints_extracted": total_constraints,
        "details": results_details
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    output_path = "/home/ianblenke/github.com/ianblenke/carnot/results/experiment_1765_eval.json"
    model_name = "unsloth/gemma-4-26B-A4B-it-GGUF"
    run_experiment(output_path, model_name)
    print(f"Experiment completed. Results saved to {output_path}")
