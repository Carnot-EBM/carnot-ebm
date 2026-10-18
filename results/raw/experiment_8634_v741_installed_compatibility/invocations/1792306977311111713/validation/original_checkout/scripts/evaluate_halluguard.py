import json
import time
import os
from carnot.verify.tier0s_halluguard import Tier0sVerifier
from carnot.verify.semantic_energy import binary_auroc

def main():
    start_time = time.time()
    
    # Generate Synthetic Corpus
    corpus_type = "synthetic"
    labels = []
    responses = []
    
    # 50 valid examples
    for i in range(50):
        # Slightly varying numbers just to be robust, but instructions just say use exact strings:
        responses.append("The sum of 2+3 is 5. Therefore the answer is 5.")
        labels.append(0)
        
    # 50 hallucinated examples
    for i in range(50):
        responses.append("The sum of 2+3 is 6. Therefore the answer is 7.")
        labels.append(1)
        
    n_eval = len(labels)
    
    # Prototype the verifier
    verifier = Tier0sVerifier(threshold=0.5)
    
    # Evaluate
    scores = [verifier.halluguard_ntk_score(r) for r in responses]
    auroc = binary_auroc(labels, scores)
    
    tier0s_viable = bool(auroc > 0.70)
    
    duration_s = time.time() - start_time
    
    result = {
        "honest_verdict": "completed",
        "tier0s_auroc": float(auroc),
        "tier0s_viable": tier0s_viable,
        "n_eval": n_eval,
        "corpus_type": corpus_type,
        "methodology_note": "Used an arithmetic deviation and sentence value delta heuristic to approximate NTK logprob variance and semantic jump magnitude on synthetic textual numbers.",
        "preconditions_checked": ["carnot.verify imports"],
        "duration_s": duration_s,
        "random_seed": 42,
        "corpus_construction_completed": True
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_2522_halluguard_tier0s.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"AUROC: {auroc:.3f}, Viable: {tier0s_viable}")
    
if __name__ == "__main__":
    main()
