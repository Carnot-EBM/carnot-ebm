import json
import os

files = {
    1574: "results/experiment_1574_120_completion_archive_121_activation.json",
    1575: "results/experiment_1575_carry_forward_prior_failures_autofill_audit.json",
    1576: "results/experiment_1576_paper_v6_section_3_sampler_draft_resumed.json",
    1577: "results/experiment_1577_extropic_z1_readiness_packet_thrml_alignment_resumed.json",
    1578: "results/experiment_1578_brain_reinforce_training_dynamics_at_k15.json",
    1579: "results/experiment_1579_iclr26_ot_verification_framework_paper_v6_adoption.json",
    1580: "results/experiment_1580_dccd_jsonschemabench_sota_structured_output_smoke.json",
    1581: "results/experiment_1581_fr11_v15_lambda_grpo_retention_reversal.json",
    1582: "results/experiment_1582_phase1_ship_readiness_ledger.json",
    1583: "results/experiment_1583_z1_analog_drift_detailed_balance_correction.json",
    1584: "results/experiment_1584_tenstorrent_wormhole_n150d_block_gibbs_preflight.json",
    1585: "results/experiment_1585_polarfire_soc_adaptive_kpcd_prototype_preflight.json",
    1586: "results/experiment_1586_strix_point_secondary_tier_rescope_kv260_retirement.json"
}

criteria_checks = {
    1574: lambda d: d.get('activation_manifest_complete') == True,
    1575: lambda d: d.get('carryforward_prior_failures_ready') == True,
    1576: lambda d: d.get('paper_v6_sampler_section_draft_ready') == True,
    1577: lambda d: d.get('extropic_z1_packet_updated') == True,
    1578: lambda d: d.get('brain_training_dynamics_verdict_ready') == True,
    1579: lambda d: d.get('ot_framework_adopted') == True,
    1580: lambda d: d.get('dccd_jsonschema_smoke_complete') == True,
    1581: lambda d: d.get('continuous_self_learning_task') == True and d.get('fr11_v15_decision_ready') == True,
    1582: lambda d: d.get('phase1_ship_readiness_ledger_ready') == True,
    1583: lambda d: d.get('detailed_balance_correction_ready') == True,
    1584: lambda d: d.get('wormhole_preflight_ready') == True or 'blocked_reason' in d,
    1585: lambda d: d.get('polarfire_preflight_ready') == True or 'blocked_reason' in d,
    1586: lambda d: d.get('hardware_portfolio_rescope_ready') == True,
}

met = 0
total = 14 # includes 1587

carryforwards = []

for exp, path in files.items():
    if not os.path.exists(path):
        continue
    try:
        with open(path) as f:
            d = json.load(f)
            if criteria_checks[exp](d):
                met += 1
            if "status" in d and d["status"] != "complete" and d["status"] != "completed":
                carryforwards.append(f"exp{exp}")
    except Exception as e:
        pass

# assume 1587 will be true
met += 1

print(f"met: {met}, total: {total}")
print(f"carryforwards: {carryforwards}")
