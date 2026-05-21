import json
import math
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from carnot.metrics.spilled_energy import compute_spilled_energy, compute_marginalized_energy

manifest_path = 'results/live_sota_balanced_telemetry_manifest_1480.jsonl'
output_json = 'results/experiment_2497_phase4_spilled_energy.json'
scores_json = 'results/experiment_2497_spilled_energy_scores.json'

spilled_energy_array = []
marginalized_energy_array = []
label_array = []
spilled_energy_scores = {}

with open(manifest_path, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        entry = json.loads(line)
        logprobs = entry.get('token_logprobs')
        if not logprobs:
            logprobs = entry.get('logprob') or entry.get('log_prob') or entry.get('top_logprobs')
        if not logprobs:
            continue
        
        spilled_energy_entry = compute_spilled_energy(logprobs)
        marginalized_energy_entry = compute_marginalized_energy(logprobs)
        
        c_label = entry.get('correctness_label', '')
        if c_label == 'correct' or entry.get('correct') is True:
            label = 0
        else:
            label = 1
            
        spilled_energy_array.append(spilled_energy_entry)
        marginalized_energy_array.append(marginalized_energy_entry)
        label_array.append(label)
        
        if 'case_id' in entry:
            spilled_energy_scores[entry['case_id']] = spilled_energy_entry

pearson_spilled, p_spilled = pearsonr(spilled_energy_array, label_array)
pearson_marginalized, p_marginalized = pearsonr(marginalized_energy_array, label_array)
auroc_spilled = roc_auc_score(label_array, spilled_energy_array)
auroc_marginalized = roc_auc_score(label_array, marginalized_energy_array)

phase4_validated_via_spilled = bool((pearson_spilled > 0.3) and (auroc_spilled > 0.65))
tier0q_viable = bool(auroc_spilled > 0.65)

deliverable = {
    "pearson_spilled": float(pearson_spilled),
    "auroc_spilled": float(auroc_spilled),
    "phase4_validated_via_spilled": phase4_validated_via_spilled,
    "tier0q_viable": tier0q_viable,
    "honest_verdict": f"complete: phase4_validated={phase4_validated_via_spilled} pearson_spilled={pearson_spilled:.4f} auroc={auroc_spilled:.4f}"
}

with open(output_json, 'w') as f:
    json.dump(deliverable, f, indent=2)

if tier0q_viable:
    with open(scores_json, 'w') as f:
        json.dump(spilled_energy_scores, f, indent=2)

print(json.dumps(deliverable, indent=2))
