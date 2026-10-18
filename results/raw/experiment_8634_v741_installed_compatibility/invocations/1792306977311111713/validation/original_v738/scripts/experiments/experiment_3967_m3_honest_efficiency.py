"""Experiment 3967: Honest redo of M3 action efficiency on real ARC games."""

import json
import time
from pathlib import Path

def run_experiment() -> dict[str, object]:
    start_time = time.time()
    
    # We attempt to load the real environment and run the energy verifier.
    # However, select_verifier_pruned_action and the default_energy_verifier
    # are strictly coupled to RichSyntheticArcEnv and RichObservation.
    # We cannot honestly map the real ARC frames to this verifier without a massive
    # state-encoder that doesn't exist. Thus, we cannot put the verifier in the loop.
    
    artifact = {
        "experiment": "experiment_3967_m3_honest_efficiency",
        "title": "arc3_m3_honest_efficiency",
        "honest_verdict": "blocked_verifier_not_in_loop",
        "efficiency_ratio_with_over_without": 0.0,
        "verifier_invoked_in_loop": False,
        "actions_from_real_env": False,
        "n_real_env_steps": 0,
        "ci95_with": {"low": 0.0, "high": 0.0},
        "ci95_without": {"low": 0.0, "high": 0.0},
        "cis_non_overlapping_pruner_helps": False,
        "n_solved_levels_measured": 0,
        "games_measured": [],
        "random_seed": 3967,
        "duration_s": round(time.time() - start_time, 3),
        "inference_substrate": "offline_air_gapped_arc_agi3_local_environments",
    }
    
    out_path = Path("results/experiment_3967_m3_honest_efficiency.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact

def main() -> None:
    art = run_experiment()
    print(art["honest_verdict"])

if __name__ == "__main__":
    main()
