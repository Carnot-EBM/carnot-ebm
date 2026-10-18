import json
import time
import hashlib
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    start_time = time.time()
    
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import publication_gate
    
    # Get publication gate status
    gate_status = publication_gate.evaluate()
    g1 = gate_status["gates"]["G1"]["pass"]
    g2 = gate_status["gates"]["G2"]["pass"]
    g3 = gate_status["gates"]["G3"]["pass"]
    g4 = gate_status["gates"]["G4"]["pass"]
    paper_ready = gate_status["paper_ready"]
    unmet_gates = gate_status.get("unmet_gates", [])
    
    # Hardcode the scope based on the prompt instructions and summarize_artifact output
    scope = "math_only_domain_bound_paper_claim_scoped"
    
    paper_ready_str = str(paper_ready).lower()
    honest_verdict = f"complete: g_gate_synthesis_v329_paper_ready_{paper_ready_str}_verifier_generalization_{scope}"
    
    # We must exclude flagged adversarial artifacts (3574)
    # Include 3573, 3575, 3576, and the FoVer dual condition artifact if available
    cited_upstream_artifacts = [
        "experiment_3573_verifier_code_bug_error_detection.json",
        "experiment_3575_verifier_discriminating_value.json",
        "experiment_3576_verifier_cross_domain_synthesis.json",
    ]
    
    d, path = publication_gate._find_headline_artifact()
    if path:
        cited_upstream_artifacts.append(Path(path).name)
        
    duration_s = time.time() - start_time
    
    payload = {
        "honest_verdict": honest_verdict,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "paper_ready": paper_ready,
        "unmet_gates": unmet_gates,
        "verifier_generalization_scope": scope,
        "p01_status": "honest-negative",
        "cited_upstream_artifacts": cited_upstream_artifacts,
        "random_seed": 42,
        "duration_s": duration_s
    }
    
    # Checksum computation
    sorted_keys = sorted(payload.keys())
    hash_str = "".join(f"{k}:{payload[k]}" for k in sorted_keys)
    payload["reproducibility_checksum"] = hashlib.md5(hash_str.encode()).hexdigest()
    
    out_file = PROJECT_ROOT / "results" / "experiment_3581_g_gate_status_synthesis_v329.json"
    out_file.write_text(json.dumps(payload, indent=2))
    
    print(f"Successfully wrote {out_file}")

if __name__ == "__main__":
    main()
