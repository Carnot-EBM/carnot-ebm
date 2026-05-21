import json
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from carnot.verify.tier0g_semantic_energy import SemanticEnergyVerifier
import os

def compute_ece(labels, probabilities, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_limits[i]
        bin_upper = bin_limits[i+1]
        
        if i == n_bins - 1:
            in_bin = (probabilities >= bin_lower) & (probabilities <= bin_upper)
        else:
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_prob_in_bin = np.mean(probabilities[in_bin])
            ece += np.abs(avg_prob_in_bin - accuracy_in_bin) * prop_in_bin
            
    return float(ece)

def main():
    start_time = time.time()
    
    # Preconditions check
    preconditions_checked = []
    
    try:
        import sklearn
        sklearn_avail = True
    except ImportError:
        sklearn_avail = False
    preconditions_checked.append({
        "resource": "sklearn",
        "available": sklearn_avail,
        "check": "import sklearn"
    })
    
    fover_lines = 0
    if os.path.exists("data/fover_corpus.jsonl"):
        with open("data/fover_corpus.jsonl", "r") as f:
            fover_lines = sum(1 for _ in f)
    preconditions_checked.append({
        "resource": "FoVer corpus",
        "available": fover_lines > 0,
        "check": "wc -l data/fover_corpus.jsonl"
    })
    
    qwen_cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF")
    qwen_cached = os.path.exists(qwen_cache_path)
    preconditions_checked.append({
        "resource": "Qwen3.6 GGUF cache",
        "available": qwen_cached,
        "check": "ls ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF"
    })
    
    if not sklearn_avail:
        with open("results/experiment_2731_semantic_energy_tier0g.json", "w") as f:
            json.dump({"honest_verdict": "blocked_sklearn_missing"}, f)
        return
        
    if fover_lines == 0:
        with open("results/experiment_2731_semantic_energy_tier0g.json", "w") as f:
            json.dump({"honest_verdict": "blocked_fover_corpus_missing"}, f)
        return

    # Load the SemanticEnergyVerifier
    verifier = SemanticEnergyVerifier(corpus_path="data/fover_corpus.jsonl", max_features=5000, random_seed=42)
    n_corpus_entries = verifier.n_corpus_entries
    n_clusters = verifier.n_clusters
    
    # 3. Evaluate on FoVer eval split
    eval_data = []
    with open("data/fover_corpus.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            eval_data.append(data)
            
    # Need determinism
    np.random.seed(42)
    _, test_data = train_test_split(eval_data, test_size=0.2, random_state=42)
    
    eval_labels = []
    eval_energies = []
    for item in test_data:
        question = item.get("question_id", "")
        response = item.get("step_text", "")
        label = item.get("label", "")
        
        energy = verifier.compute_energy(question, response)
        
        eval_energies.append(energy)
        # label=1 if incorrect (hallucination), label=0 if correct
        eval_labels.append(1 if label == "incorrect" else 0)
        
    eval_energies = np.array(eval_energies)
    eval_labels = np.array(eval_labels)
    
    tier0g_auroc = roc_auc_score(eval_labels, eval_energies)
    
    # Normalizing energies to [0,1] for ECE? Or just pass as probabilities?
    # Energy is usually in [0, \infty]. To compute ECE, we can convert energy to probability: 
    # P(hallucination) = 1 - exp(-energy) ? Or maybe just normalize.
    # The paper says: ECE calculation
    probs = 1.0 - np.exp(-eval_energies) # probability of incorrect
    tier0g_ece = compute_ece(eval_labels, probs, 10)
    
    energy_std = np.std(eval_energies)
    energy_max = np.max(eval_energies)
    energy_min = np.min(eval_energies)
    
    tier0g_non_degenerate = bool(energy_std > 0.01 and (energy_max - energy_min) > 0.1)
    
    # 4. Adversarial check: run on 3 synthetic Qwen3.6-format responses
    synthetic_responses = [
        "<think>\nThis is a test.\n</think>\nThe answer is 42.",
        "```python\ndef solve():\n    return 'hello'\n```\nHere is the code.",
        "**Step 1:** Calculate distance.\nDistance = 50 * 2 = 100 miles."
    ]
    synthetic_energies = [verifier.compute_energy("", r) for r in synthetic_responses]
    all_non_zero = all(e > 0.0 for e in synthetic_energies)
    
    # 5. If qwen_cached: read 3 cached GGUF output examples
    live_gguf_tested = False
    gguf_energies = []
    gguf_non_degenerate = False
    
    if qwen_cached:
        # read from results/arm_ebm_logprob_telemetry_manifest_1556.jsonl
        telemetry_file = "results/arm_ebm_logprob_telemetry_manifest_1556.jsonl"
        if os.path.exists(telemetry_file):
            with open(telemetry_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "response_text" in data and data["response_text"]:
                            gguf_energies.append(verifier.compute_energy("", data["response_text"]))
                            if len(gguf_energies) == 3:
                                break
                    except:
                        pass
        
        if len(gguf_energies) == 3:
            live_gguf_tested = True
            g_std = np.std(gguf_energies)
            g_max = np.max(gguf_energies)
            g_min = np.min(gguf_energies)
            gguf_non_degenerate = bool(g_std > 0.001 and (g_max - g_min) > 0.01)

    duration_s = time.time() - start_time
    
    result = {
        "honest_verdict": "complete: Semantic Energy tier0g implemented and evaluated",
        "tier0g_auroc": float(tier0g_auroc),
        "tier0g_non_degenerate": tier0g_non_degenerate,
        "synthetic_energies": synthetic_energies,
        "all_non_zero": all_non_zero,
        "tier0g_ece": float(tier0g_ece),
        "n_clusters": n_clusters,
        "module_created": True,
        "random_seed": 42,
        "duration_s": duration_s,
        "preconditions_checked": preconditions_checked
    }
    
    if qwen_cached:
        result["live_gguf_tested"] = live_gguf_tested
        result["gguf_energies"] = gguf_energies
        result["gguf_non_degenerate"] = gguf_non_degenerate
        
    with open("results/experiment_2731_semantic_energy_tier0g.json", "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"AUROC: {tier0g_auroc:.4f}")
    print(f"Non-degenerate: {tier0g_non_degenerate}")
    print(f"Clusters: {n_clusters}")
    print(f"Synthetic: {synthetic_energies}")
    if qwen_cached:
        print(f"GGUF: {gguf_energies}")

if __name__ == "__main__":
    main()
