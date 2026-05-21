import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

from carnot.verify.hallufield_verifier import HalluFieldVerifier
from carnot.verify.semantic_energy import binary_auroc
from carnot.verify.halt_probe import _read_jsonl, label_from_entry, oof_halt_risk_scores, _preconditions
from carnot.verify.pcib_verifier import evaluate_pcib

def evaluate_hallufield():
    start_time = time.perf_counter()
    manifest_path = Path("results/live_sota_balanced_telemetry_manifest_1480.jsonl")
    
    preconditions = _preconditions(manifest_path)
    
    entries = _read_jsonl(manifest_path, limit=36)
    labels = [label_from_entry(e) for e in entries]
    
    # 1. HalluField
    verifier = HalluFieldVerifier()
    hallufield_scores = []
    for entry in entries:
        logprobs = entry.get("token_logprobs") or []
        score = verifier.score(logprobs)
        hallufield_scores.append(score)
        
    auroc = binary_auroc(labels, hallufield_scores)
    
    # 2. Orthogonality
    _, _, pcib_scores = evaluate_pcib(entries, labels, random_seed=42)
    halt_scores = oof_halt_risk_scores(entries, labels, random_seed=42)
    
    r_pcib, _ = pearsonr(hallufield_scores, pcib_scores)
    r_halt, _ = pearsonr(hallufield_scores, halt_scores)
    
    duration = time.perf_counter() - start_time
    
    # Construct Results
    result = {
        "honest_verdict": f"complete: HalluFieldVerifier Tier 0m evaluated with AUROC {auroc:.4f}",
        "hallufield_auroc": auroc,
        "hallufield_vs_semantic_energy_delta": auroc - 0.810,
        "orthogonality_vs_pcib": r_pcib,
        "orthogonality_vs_halt": r_halt,
        "temperature_grid_used": verifier.temp_grid,
        "n_eval_examples": 36,
        "random_seed": 42,
        "duration_s": duration,
        "preconditions_checked": preconditions,
    }
    
    with open("results/experiment_2449_hallufield_tier0m.json", "w") as f:
        json.dump(result, f, indent=2)
        
    # Construct Scores for conformal ensemble
    scores_result = {
        "verifier": "hallufield",
        "scores": [{"idx": i, "score": s, "label": l} for i, (s, l) in enumerate(zip(hallufield_scores, labels))]
    }
    
    with open("results/experiment_2449_tier0m_scores.json", "w") as f:
        json.dump(scores_result, f, indent=2)

if __name__ == "__main__":
    evaluate_hallufield()
