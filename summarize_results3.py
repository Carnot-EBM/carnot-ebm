import json
import os

files = [
    "results/experiment_1582_phase1_ship_readiness_ledger.json",
    "results/experiment_1578_brain_reinforce_training_dynamics_at_k15.json",
    "results/experiment_1581_fr11_v15_lambda_grpo_retention_reversal.json",
    "results/experiment_1579_iclr26_ot_verification_framework_paper_v6_adoption.json",
    "results/experiment_1580_dccd_jsonschemabench_sota_structured_output_smoke.json",
    "results/experiment_1584_tenstorrent_wormhole_n150d_block_gibbs_preflight.json",
    "results/experiment_1585_polarfire_soc_adaptive_kpcd_prototype_preflight.json",
    "results/experiment_1586_strix_point_secondary_tier_rescope_kv260_retirement.json"
]

for f in files:
    if os.path.exists(f):
        try:
            with open(f) as fp:
                d = json.load(fp)
                print(f"\n--- {f} ---")
                for k in ["status", "honest_verdict", "blocked_reason", "blocking_items"]:
                    if k in d:
                        print(f"{k}: {d[k]}")
        except Exception:
            pass
