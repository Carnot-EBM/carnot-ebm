import json
import os
import shutil

def run_retro():
    """Generates the operational retrospective for milestone 2026.05.242."""
    capstone_path = "results/experiment_2516_capstone_v242.json"
    with open(capstone_path, "r") as f:
        capstone = json.load(f)
        
    n_completed = capstone.get("n_experiments_completed", 0)
    best_242_auroc = capstone.get("best_242_auroc")
    phase4_validated_any = capstone.get("phase4_validated_any")
    arxiv_ready = capstone.get("arxiv_ready")
    
    top_3_successes = capstone.get("top_3_successes", [])
    if not top_3_successes and "synthesis" in capstone and "top_3_successes" in capstone["synthesis"]:
        top_3_successes = capstone["synthesis"]["top_3_successes"]

    top_3_gaps_for_243 = capstone.get("top_3_gaps_for_243", [])
    if not top_3_gaps_for_243 and "synthesis" in capstone and "top_3_gaps_for_243" in capstone["synthesis"]:
        top_3_gaps_for_243 = capstone["synthesis"]["top_3_gaps_for_243"]
    
    honest_verdict = f"complete: best_242_auroc={best_242_auroc}; phase4_validated_any={phase4_validated_any}; arxiv_ready={arxiv_ready}"
    
    retro_data = {
        "schema": "carnot.operational_retro.v66",
        "milestone": "2026.05.242",
        "n_experiments_completed": n_completed,
        "best_242_auroc": best_242_auroc,
        "phase4_validated_any": phase4_validated_any,
        "arxiv_ready": arxiv_ready,
        "top_3_successes": top_3_successes,
        "top_3_gaps_for_243": top_3_gaps_for_243,
        "honest_verdict": honest_verdict
    }
    
    deliverable_path = "results/experiment_2517_retro_v242.json"
    with open(deliverable_path, "w") as f:
        json.dump(retro_data, f, indent=2)
        
    # Copy to the operational retro location for consistency
    shutil.copy(deliverable_path, "results/operational_retro_2026_05_242.json")
    
    return retro_data

if __name__ == "__main__":
    run_retro()
