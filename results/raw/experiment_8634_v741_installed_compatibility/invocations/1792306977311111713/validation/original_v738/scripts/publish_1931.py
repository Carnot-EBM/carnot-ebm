import sys
import json
from pathlib import Path

# Add python directory to path so carnot can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from carnot.pipeline.hf_publisher import HuggingFacePublisher

def main():
    # Find a checkpoint < 50MB
    results_dir = Path("results")
    candidates = list(results_dir.glob("*.safetensors"))
    artifact_path = None
    for cand in candidates:
        if cand.stat().st_size < 50 * 1024 * 1024:
            artifact_path = cand
            break
            
    if not artifact_path:
        print("No valid artifact found, running publisher with dummy path to get blocked verdict.")
        artifact_path = Path("dummy.safetensors")

    publisher = HuggingFacePublisher(str(artifact_path))
    deliverable = publisher.run_publish()

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "experiment_1931_huggingface_mirror.json"
    
    with open(out_path, "w") as f:
        json.dump(deliverable, f, indent=2)
        
    print(f"Wrote deliverable to {out_path}")

if __name__ == "__main__":
    main()
