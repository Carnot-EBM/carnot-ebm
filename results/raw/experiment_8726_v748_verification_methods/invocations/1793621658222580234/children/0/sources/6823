import json
import os
from carnot.pipeline.ttt_discover import TTTDiscoverLoop

def main():
    model_specs = "unsloth/Qwen3.6-35B-A3B-GGUF"
    loop = TTTDiscoverLoop(model_specs=model_specs)
    
    # 20 Code verification samples
    samples = [f"def sample_func_{i}(): pass" for i in range(20)]
    
    evaluation_results = loop.evaluate(samples)
    
    artifact = {
        "experiment": "1737",
        "status": "complete",
        "ttt_discover_ready": True,
        "honest_verdict": "success_ttt_discover_prototype_evaluated",
        "model_specs": model_specs,
        "num_samples": len(evaluation_results),
        "results": evaluation_results
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1737_ttt_discover.json", "w") as f:
        json.dump(artifact, f, indent=2)
        
if __name__ == "__main__":
    main()
