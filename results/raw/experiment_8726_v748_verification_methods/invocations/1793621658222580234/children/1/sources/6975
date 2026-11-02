import json
import os

def generate_retro_data() -> dict:
    """Generate the retrospective analysis data for Milestone 211."""
    return {
        "experiment_id": 2131,
        "schema": "carnot.milestone_retro.v1",
        "milestone": "2026.05.211",
        "milestone_title": "CARM, AdamFLIP, and Gate Blocks",
        "run_date": "20260517",
        "status": "complete",
        "completed_task_count": 4,
        "blocked_task_count": 3,
        "failed_task_count": 0,
        "completed_experiments": [2120, 2121, 2122, 2127],
        "blocked_experiments": [2125, 2128, 2129],
        "highest_leverage_actions": [
            "Implemented and wired CARM into the LLM solver (Exp 2121, 2122).",
            "Successfully integrated AdamFLIP into the CSL loop (Exp 2127)."
        ],
        "estimated_time_savings_pct": 10.0,
        "meta_reflection": "Progress was made on CARM and AdamFLIP integrations. However, multiple tasks (2125, 2128, 2129) were blocked at the pre-gate layer due to missing prior_failures, highlighting the need for stricter rerun discipline.",
        "honest_verdict": "complete: carm_and_adamflip_integrated_but_multiple_gate_blocks"
    }

def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    retro_data = generate_retro_data()
    
    out_path = os.path.join(results_dir, "experiment_2131_retro.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(retro_data, f, indent=2)
        
    print(f"Wrote retro to {out_path}")

if __name__ == "__main__":
    main()
