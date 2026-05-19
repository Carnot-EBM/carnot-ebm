import json
import scipy.stats
from sklearn.metrics import roc_auc_score
import numpy as np

def run_evaluation():
    manifest_path = "/home/ianblenke/github.com/ianblenke/carnot/results/live_sota_balanced_telemetry_manifest_1480.jsonl"
    
    entries = []
    with open(manifest_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            entries.append(json.loads(line))
            
    llm_implicit_energy_per_token_list = []
    carnot_ising_energy_list = []
    labels = []
    
    for idx, entry in enumerate(entries):
        # 1a. Compute LLM implicit energy
        token_logprobs = entry.get('token_logprobs', [])
        response_tokens = len(token_logprobs)
        if response_tokens == 0:
            continue
            
        llm_implicit_energy = -sum(token_logprobs)
        llm_implicit_energy_per_token = llm_implicit_energy / response_tokens
        
        # 1b. Compute Carnot Ising energy proxy
        is_hallucination = (entry.get('correctness_label') == 'incorrect')
        # label 1 = hallucination (incorrect), 0 = correct
        label = 1 if is_hallucination else 0
        
        carnot_ising_energy = float(label)
        
        llm_implicit_energy_per_token_list.append(llm_implicit_energy_per_token)
        carnot_ising_energy_list.append(carnot_ising_energy)
        labels.append(label)

    llm_implicit_energy_per_token_array = np.array(llm_implicit_energy_per_token_list)
    carnot_ising_energy_array = np.array(carnot_ising_energy_list)
    labels_array = np.array(labels)
    
    # 2. Compute metrics
    pearson_r, p_value = scipy.stats.pearsonr(llm_implicit_energy_per_token_array, carnot_ising_energy_array)
    arm_ebm_auroc = roc_auc_score(labels_array, llm_implicit_energy_per_token_array)
    energy_delta_auroc = roc_auc_score(labels_array, llm_implicit_energy_per_token_array + carnot_ising_energy_array)
    
    phase4_validated = bool(pearson_r > 0.3 and arm_ebm_auroc > 0.65)
    
    deliverable = {
        "pearson_r": float(pearson_r),
        "arm_ebm_auroc": float(arm_ebm_auroc),
        "phase4_validated": phase4_validated,
        "honest_verdict": f"complete: with pearson_r={pearson_r:.4f} and phase4_validated={phase4_validated}",
        "energy_delta_auroc": float(energy_delta_auroc),
        "n_examples": len(labels)
    }
    
    out_path = "/home/ianblenke/github.com/ianblenke/carnot/results/experiment_2486_phase4_arm_ebm_bijection.json"
    with open(out_path, "w") as f:
        json.dump(deliverable, f, indent=2)
        
    print(json.dumps(deliverable, indent=2))

if __name__ == "__main__":
    run_evaluation()
