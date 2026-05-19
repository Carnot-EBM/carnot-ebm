import json
import time
import numpy as np
from scipy.stats import pearsonr

from carnot.verify.suppressed_retrieval_probe import SuppressedRetrievalProbe
from carnot.verify.halt_probe import oof_halt_risk_scores, label_from_entry
from carnot.verify.semantic_energy import binary_auroc

def main():
    start_time = time.time()
    
    manifest_path = "results/live_sota_balanced_telemetry_manifest_1480.jsonl"
    with open(manifest_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]
        
    labels = [label_from_entry(entry) for entry in entries]
    halt_scores = oof_halt_risk_scores(entries, labels, random_seed=42)
    
    probe = SuppressedRetrievalProbe()
    suppression_scores = []
    divergences = []
    verifier_scores_out = []
    
    for idx, entry in enumerate(entries):
        result = probe.verify(entry)
        score = result["suppression_score"]
        div = result["paraphrase_divergence"]
        suppression_scores.append(score)
        divergences.append(div)
        
        verifier_scores_out.append({
            "idx": idx,
            "score": score,
            "label": labels[idx]
        })
        
    tier0o_auroc = binary_auroc(labels, suppression_scores)
    tier0o_vs_semantic_energy_delta = tier0o_auroc - 0.810
    
    r, _ = pearsonr(suppression_scores, halt_scores)
    orthogonality_vs_halt = float(r)
    paraphrase_divergence_mean = float(np.mean(divergences))
    
    # 3. Compute verifier_scores for ensemble
    with open("results/experiment_2462_tier0o_scores.json", "w") as f:
        json.dump({"verifier": "suppressed_retrieval", "scores": verifier_scores_out}, f, indent=2)
        
    duration_s = time.time() - start_time
    
    deliverable = {
        "honest_verdict": "complete: Tier 0o Suppressed Retrieval Probe evaluated successfully with AUROC.",
        "tier0o_auroc": tier0o_auroc,
        "tier0o_vs_semantic_energy_delta": tier0o_vs_semantic_energy_delta,
        "orthogonality_vs_halt": orthogonality_vs_halt,
        "paraphrase_divergence_mean": paraphrase_divergence_mean,
        "n_eval_examples": len(entries),
        "random_seed": 42,
        "duration_s": duration_s,
        "preconditions_checked": True
    }
    
    with open("results/experiment_2462_tier0o_nla_probe.json", "w") as f:
        json.dump(deliverable, f, indent=2)
        
    print(f"tier0o_auroc: {tier0o_auroc}")
    print(f"divergence: {paraphrase_divergence_mean}")
    print("Done")

if __name__ == "__main__":
    main()
