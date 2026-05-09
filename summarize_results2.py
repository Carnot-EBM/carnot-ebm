import json
import os
import glob

files = [
    "results/experiment_1574_120_completion_archive_121_activation.json",
    "results/experiment_1575_carry_forward_prior_failures_autofill_audit.json",
    "results/experiment_1576_paper_v6_section_3_sampler_draft_resumed.json",
    "results/experiment_1577_extropic_z1_readiness_packet_thrml_alignment_resumed.json",
    "results/experiment_1578_brain_reinforce_training_dynamics_at_k15.json",
    "results/experiment_1579_iclr26_ot_verification_framework_paper_v6_adoption.json",
    "results/experiment_1580_dccd_jsonschemabench_sota_structured_output_smoke.json",
    "results/experiment_1581_fr11_v15_lambda_grpo_retention_reversal.json",
    "results/experiment_1582_phase1_ship_readiness_ledger.json",
    "results/experiment_1583_z1_analog_drift_detailed_balance_correction.json",
    "results/experiment_1584_tenstorrent_wormhole_n150d_block_gibbs_preflight.json",
    "results/experiment_1585_polarfire_soc_adaptive_kpcd_prototype_preflight.json",
    "results/experiment_1586_strix_point_secondary_tier_rescope_kv260_retirement.json"
]

for f in files:
    if os.path.exists(f):
        try:
            with open(f) as fp:
                d = json.load(fp)
                print(f"--- {f} ---")
                print("KEYS:", list(d.keys()))
                if "outcome" in d: print("outcome:", d["outcome"])
                if "results" in d: print("results:", d["results"])
                if "blockers" in d: print("blockers:", d["blockers"])
                if "carry_forwards" in d: print("carry_forwards:", d["carry_forwards"])
                if "next_steps" in d: print("next_steps:", d["next_steps"])
        except Exception:
            pass
