import json
import os
import sys
import numpy as np
from pathlib import Path

# Add carnot to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from carnot.pipeline.nup_metric import NUPMetric

def generate_mock_gradients(is_hallucination: bool, steps: int = 10) -> list[np.ndarray]:
    """Generate mock gradients for a trajectory."""
    np.random.seed(42 if is_hallucination else 24)
    gradients = []
    
    # baseline noise
    base_grad = np.random.randn(128)
    for i in range(steps):
        if is_hallucination and i == steps // 2:
            # Sudden shift in gradient direction
            grad = -base_grad + np.random.randn(128) * 0.1
        else:
            grad = base_grad + np.random.randn(128) * 0.1
        gradients.append(grad)
        
    return gradients

def main():
    print("Starting experiment 3405: NUP Metric Evaluation")
    
    # Config
    MODEL_SPECS = ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    threshold = 1.2
    n_samples = 100
    
    metric = NUPMetric(threshold=threshold)
    
    # We use synthetic data for CI runs
    # 50 correct, 50 hallucinations
    ground_truth = [False] * 50 + [True] * 50
    
    results = []
    correct_detections = 0
    
    for i, is_hal in enumerate(ground_truth):
        gradients = generate_mock_gradients(is_hallucination=is_hal, steps=10)
        
        # Evaluate using NUP metric
        eval_result = metric.evaluate(gradients)
        detected = eval_result["hallucination_detected"]
        
        if detected == is_hal:
            correct_detections += 1
            
        results.append({
            "sample_id": i,
            "ground_truth_hallucination": is_hal,
            "detected_hallucination": detected,
            "energies": eval_result["energies"]
        })
        
    accuracy = correct_detections / n_samples
    
    artifact = {
        "experiment": 3405,
        "schema": "carnot.nup_metric_evaluation.v1",
        "model_specs": MODEL_SPECS,
        "threshold": threshold,
        "n_samples": n_samples,
        "accuracy": accuracy,
        "inference_mode": "simulated",
        "results": results
    }
    
    # Ensure results directory exists
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    out_path = out_dir / "experiment_3405_nup_metric_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
        
    print(f"Evaluation complete. Accuracy: {accuracy:.2f}")
    print(f"Results written to {out_path}")

if __name__ == "__main__":
    main()
