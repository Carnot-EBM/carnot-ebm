"""
Experiment 1773: CARM Evaluation on dual models.

Spec: REQ-CARM-1773-1
"""
import json
from pathlib import Path
from carnot.carm.prototype import CARMExtractor

def run_experiment(output_path="results/experiment_1773_care_evaluation.json"):
    """Evaluate extraction recall and false accept rate on dual models."""
    test_suite_path = Path("results/experiment_1771_care_test_suite.json")
    if not test_suite_path.exists():
        # Fallback or raise for actual execution
        raise FileNotFoundError(f"Test suite not found at {test_suite_path}")
        
    test_suite = json.loads(test_suite_path.read_text())
    cases = test_suite.get("cases", [])
    
    models = [
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF"
    ]
    
    total_recall_sum = 0.0
    total_fa_sum = 0.0
    
    for model_spec in models:
        extractor = CARMExtractor(model_spec=model_spec)
        
        true_positives = 0
        false_negatives = 0
        false_positives = 0
        true_negatives = 0
        
        for case in cases:
            extracted = extractor.extract_constraints(case["instruction"])
            ground_truth = case.get("ground_truth", {})
            
            # Simple evaluation logic:
            # If ground_truth is not empty, it's a positive case.
            # If it is empty, it's a negative case.
            is_positive_case = bool(ground_truth)
            
            # For this evaluation, we consider an exact match as correct.
            # A more nuanced evaluation might check specific keys.
            if is_positive_case:
                if extracted == ground_truth:
                    true_positives += 1
                else:
                    false_negatives += 1
            else:
                if extracted == ground_truth:
                    true_negatives += 1
                else:
                    false_positives += 1
        
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        # False Accept Rate (FAR) = False Positives / (False Positives + True Negatives)
        far = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
        
        total_recall_sum += recall
        total_fa_sum += far
        
        print(f"Model: {model_spec} - Recall: {recall:.2f}, FAR: {far:.2f}")

    avg_recall = total_recall_sum / len(models) if models else 0.0
    avg_far = total_fa_sum / len(models) if models else 0.0
    
    deliverable = {
        "schema": "carnot.carm.evaluation.v1",
        "experiment_id": 1773,
        "models_evaluated": models,
        "recall_rate": float(avg_recall),
        "false_accept_rate": float(avg_far),
        "status": "complete",
        "honest_verdict": "complete: CARM dual-model evaluated",
    }
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deliverable, indent=2))
    
    print(f"Overall Recall: {avg_recall:.2f}, Overall FAR: {avg_far:.2f}")
    return deliverable

if __name__ == "__main__":
    run_experiment()
