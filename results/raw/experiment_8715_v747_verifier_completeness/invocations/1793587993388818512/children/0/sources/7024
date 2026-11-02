import json
import os
import shutil


def run_retro():
    """Generates the operational retrospective for milestone 2026.05.243."""
    capstone_path = "results/experiment_2528_capstone_v243.json"
    with open(capstone_path, "r") as f:
        capstone = json.load(f)

    n_completed = capstone.get("n_experiments_completed", 0)
    best_243_auroc = capstone.get("best_243_auroc")
    phase4_final_status = capstone.get("phase4_final_status")
    arxiv_ready = capstone.get("arxiv_ready")
    operator_recommendation = capstone.get("operator_recommendation")

    top_3_successes = capstone.get("top_3_successes", [])
    if (
        not top_3_successes
        and "synthesis" in capstone
        and "top_3_successes" in capstone["synthesis"]
    ):
        top_3_successes = capstone["synthesis"]["top_3_successes"]

    top_3_gaps_for_244 = capstone.get("top_3_gaps_for_244", [])
    if (
        not top_3_gaps_for_244
        and "synthesis" in capstone
        and "top_3_gaps_for_244" in capstone["synthesis"]
    ):
        top_3_gaps_for_244 = capstone["synthesis"]["top_3_gaps_for_244"]

    honest_verdict = f"complete: best_243_auroc={best_243_auroc}; phase4_final_status={phase4_final_status}; arxiv_ready={arxiv_ready}"

    retro_data = {
        "honest_verdict": honest_verdict,
        "schema": "carnot.operational_retro.v67",
        "n_experiments_completed": n_completed,
        "best_243_auroc": best_243_auroc,
        "phase4_final_status": phase4_final_status,
        "arxiv_ready": arxiv_ready,
        "operator_recommendation": operator_recommendation,
        "top_3_successes": top_3_successes,
        "top_3_gaps_for_244": top_3_gaps_for_244,
    }

    deliverable_path = "results/experiment_2529_retro_v243.json"
    with open(deliverable_path, "w") as f:
        json.dump(retro_data, f, indent=2)

    # Copy to the operational retro location for consistency
    shutil.copy(deliverable_path, "results/operational_retro_2026_05_243.json")

    return retro_data


if __name__ == "__main__":
    run_retro()
