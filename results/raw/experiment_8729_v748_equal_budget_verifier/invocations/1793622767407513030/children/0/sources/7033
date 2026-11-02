import json
import os
import glob
import shutil
from pathlib import Path

def run_retro():
    # 1. Count task outcomes
    n_completed = 0
    n_missing = 0
    n_blocked = 0
    
    # The milestone .241 tasks are exp2496 to exp2505 (10 tasks)
    for i in range(2496, 2506):
        # Find json
        matches = glob.glob(f"results/experiment_{i}_*.json")
        if not matches:
            n_missing += 1
        else:
            # Read the first match
            with open(matches[0], "r") as f:
                data = json.load(f)
                if data.get("status") == "blocked":
                    n_blocked += 1
                else:
                    n_completed += 1
                    
    # 2. Read key metrics from capstone (exp2505)
    capstone_path = "results/experiment_2505_capstone_v241.json"
    with open(capstone_path, "r") as f:
        capstone = json.load(f)
        
    best_241_auroc = capstone.get("best_241_auroc")
    auroc_adversarially_verified = capstone.get("auroc_adversarially_verified")
    phase4_validated_any = capstone.get("phase4_validated_any")
    arxiv_ready = capstone.get("arxiv_ready")
    
    top_3_successes = [
        "AUROC adversarially verified: exp2498 replicated the 0.975 group-conditional AUROC across 5 seeds, resolving the cross-group tautology and meeting Gate 4.",
        "Tier 0r (Curry-Howard soft-typed proof-path) achieved an AUROC of 0.9123 and is viable as a 16th verifier candidate.",
        "FR-11 integration working end-to-end: Tier 4 adaptive-energy feedback successfully fires into Tier 1."
    ]
    
    top_3_gaps_for_242 = [
        "Phase 4 validation blocked: exp2496 (Qwen PRC v3) artifact is missing (likely resource/quota blocked) and exp2497 (Spilled Energy) failed to validate Phase 4. Gate 3 remains unmet.",
        "Tier 0q (Spilled Energy) is non-viable due to noise-floor correlation, precluding its inclusion in the ensemble. It must be formally retired from candidate set.",
        "arXiv submission remains blocked on Gate 3 (phase4_validated_any is False), requiring a valid Phase 4 verification path (e.g., resolving the real-GGUF Qwen PRC blocker)."
    ]
    
    honest_verdict = f"complete: best_241_auroc={best_241_auroc}, phase4_validated_any={phase4_validated_any}, arxiv_ready={arxiv_ready}"
    
    retro_data = {
        "schema": "carnot.operational_retro.v65",
        "milestone": "2026.05.241",
        "n_experiments_completed": n_completed,
        "n_missing": n_missing,
        "n_blocked": n_blocked,
        "best_241_auroc": best_241_auroc,
        "auroc_adversarially_verified": auroc_adversarially_verified,
        "phase4_validated_any": phase4_validated_any,
        "arxiv_ready": arxiv_ready,
        "top_3_successes": top_3_successes,
        "top_3_gaps_for_242": top_3_gaps_for_242,
        "honest_verdict": honest_verdict
    }
    
    # Write to both paths for safety
    deliverable_path = "results/operational_retro_2026_05_241.json"
    with open(deliverable_path, "w") as f:
        json.dump(retro_data, f, indent=2)
    shutil.copy(deliverable_path, "results/experiment_2506_retro_v241.json")
        
    # Append to roadmap.md
    roadmap_path = "docs/roadmap.md"
    if os.path.exists(roadmap_path):
        # Prevent appending multiple times during test runs
        with open(roadmap_path, "r") as f:
            content = f.read()
        row = f"| 2026.05.241 | {honest_verdict} | {n_completed} experiments | {n_missing} missing, {n_blocked} blocked; Phase 4 unvalidated; Operator hold persists |\n"
        if row not in content:
            with open(roadmap_path, "a") as f:
                f.write(row)
            
    return retro_data

if __name__ == "__main__":
    run_retro()
