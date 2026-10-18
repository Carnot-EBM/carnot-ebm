import json
from pathlib import Path
from carnot.hardware.polarfire_continuity_v17 import perform_continuity_check

def run():
    artifact = perform_continuity_check()
    
    output_path = Path("results/experiment_3594_polarfire_continuity_v17.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
        
    print(f"Artifact written to {output_path}")

if __name__ == "__main__":
    run()
