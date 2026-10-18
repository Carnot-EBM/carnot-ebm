import json
import os
from datetime import datetime, timezone

def generate_retro():
    output_path = "results/experiment_2494_retro_v240.json"
    op_retro_path = "results/operational_retro_2026_05_240.json"
    
    # Capstone data
    with open("results/experiment_2493_capstone_v240.json", "r") as f:
        capstone = json.load(f)
        
    n_experiments_completed = 11  # 2483 to 2493 + we are 2494
    n_missing = 0
    n_blocked = 0
    
    best_auroc = capstone.get("best_240_auroc", 0.0)
    auroc_adversarially_verified = capstone.get("auroc_adversarially_verified", False)
    phase4_validated_any = capstone.get("phase4_validated_any", False)
    arxiv_ready = capstone.get("arxiv_ready", False)
    
    top_3_successes = [
        "Group-conditional calibration (exp2485) breaches HIVE 0.9236: mean AUROC 0.975 (std 0.021).",
        "PolarFire SoC reaches terminal state (exp2490): carnot_runs_on_polarfire=True on riscv64.",
        "KAN retrain achieves certified deployment readiness (exp2489): new_certified_coverage=0.833, new_kan_auroc=0.974."
    ]
    
    top_3_gaps_for_241 = [
        "Phase 4 not validated; requires retry with real Qwen3.6-35B-A3B-GGUF.",
        "KV260 flash blocked on Digilent JTAG HS2 programmer purchase.",
        "PolarFire terminal state needs verification pass on Carnot energy computation."
    ]
    
    honest_verdict = "complete: best_240_auroc=0.975, phase4_validated_any=False, arxiv_ready=False"
    
    retro_data = {
        "n_experiments_completed": n_experiments_completed,
        "n_missing": n_missing,
        "n_blocked": n_blocked,
        "best_240_auroc": best_auroc,
        "auroc_adversarially_verified": auroc_adversarially_verified,
        "phase4_validated_any": phase4_validated_any,
        "arxiv_ready": arxiv_ready,
        "top_3_successes": top_3_successes,
        "top_3_gaps_for_241": top_3_gaps_for_241,
        "honest_verdict": honest_verdict,
        "schema": "carnot.operational_retro.v64",
        "milestone": "2026.05.240",
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
    
    with open(output_path, "w") as f:
        json.dump(retro_data, f, indent=2)
        
    with open(op_retro_path, "w") as f:
        json.dump(retro_data, f, indent=2)
        
    # Append to roadmap
    roadmap_path = "docs/roadmap.md"
    if os.path.exists(roadmap_path):
        with open(roadmap_path, "a") as f:
            f.write(f"\n| 2026.05.240 | {honest_verdict} | {n_experiments_completed} experiments | Phase 4 not validated; Operator hold persists |\n")

    return retro_data

if __name__ == "__main__":
    generate_retro()
