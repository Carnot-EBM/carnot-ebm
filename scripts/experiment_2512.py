import os
import time
import json
import numpy as np
import sys

def manual_roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_idx = y_true == 1
    neg_idx = y_true == 0
    pos_scores = y_score[pos_idx]
    neg_scores = y_score[neg_idx]
    
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
        
    # Vectorized computation of pairs
    # Count pairs where pos_score > neg_score
    diffs = pos_scores[:, None] - neg_scores[None, :]
    return float(np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)) / (len(pos_scores) * len(neg_scores))

# Add python directory to path so we can import carnot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from carnot.fr11.tier2_memory import Tier2ThresholdMemory

def main():
    start_time = time.time()
    seed = 42
    np.random.seed(seed)
    
    # 0. PRECONDITIONS
    # We implicitly verify it by successfully importing and instantiating
    memory = Tier2ThresholdMemory(db_path="data/test_fr11_tier2_memory.db")
    preconditions_checked = [
        "python3 -c 'import carnot.fr11'", 
        "FR-11 Tier 2 SQLite memory implementation instantiated"
    ]
    
    # 1. Create synthetic domain corpus: 2 domains, 64 examples each
    # Domain A: shifted high
    n_examples = 64
    n_train = 32
    n_test = 32
    
    # Domain A (factual) - Positives ~ 0.8, Negatives ~ 0.6
    domain_a_pos = np.random.normal(0.8, 0.05, n_examples // 2)
    domain_a_neg = np.random.normal(0.6, 0.05, n_examples // 2)
    domain_a_scores = np.concatenate([domain_a_pos, domain_a_neg])
    domain_a_labels = np.concatenate([np.ones(n_examples // 2), np.zeros(n_examples // 2)])
    
    # Domain B (reasoning) - Positives ~ 0.4, Negatives ~ 0.2
    domain_b_pos = np.random.normal(0.4, 0.05, n_examples // 2)
    domain_b_neg = np.random.normal(0.2, 0.05, n_examples // 2)
    domain_b_scores = np.concatenate([domain_b_pos, domain_b_neg])
    domain_b_labels = np.concatenate([np.ones(n_examples // 2), np.zeros(n_examples // 2)])
    
    # Shuffle
    idx_a = np.random.permutation(n_examples)
    domain_a_scores = domain_a_scores[idx_a]
    domain_a_labels = domain_a_labels[idx_a]
    
    idx_b = np.random.permutation(n_examples)
    domain_b_scores = domain_b_scores[idx_b]
    domain_b_labels = domain_b_labels[idx_b]
    
    # 2. Implement 32-example threshold adapter (Already in tier2_memory.py)
    # Train
    train_a_scores = domain_a_scores[:n_train]
    train_a_labels = domain_a_labels[:n_train]
    train_b_scores = domain_b_scores[:n_train]
    train_b_labels = domain_b_labels[:n_train]
    
    memory.update_domain_delta("factual", list(train_a_scores), list(train_a_labels))
    memory.update_domain_delta("reasoning", list(train_b_scores), list(train_b_labels))
    
    # 3. Validate on held-out 32 examples per domain
    test_a_scores = domain_a_scores[n_train:]
    test_a_labels = domain_a_labels[n_train:]
    test_b_scores = domain_b_scores[n_train:]
    test_b_labels = domain_b_labels[n_train:]
    
    all_test_labels = np.concatenate([test_a_labels, test_b_labels])
    all_test_raw_scores = np.concatenate([test_a_scores, test_b_scores])
    
    adapted_a_scores = [memory.apply_delta("factual", s) for s in test_a_scores]
    adapted_b_scores = [memory.apply_delta("reasoning", s) for s in test_b_scores]
    all_test_adapted_scores = np.concatenate([adapted_a_scores, adapted_b_scores])
    
    memory_augmented_auroc = manual_roc_auc_score(all_test_labels, all_test_adapted_scores)
    no_memory_baseline_auroc = manual_roc_auc_score(all_test_labels, all_test_raw_scores)
    
    # Compile results
    duration_s = time.time() - start_time
    verdict = "complete: memory_augmented_auroc >= 0.95" if memory_augmented_auroc >= 0.95 else "failed: auroc too low"
    
    result = {
        "honest_verdict": verdict,
        "memory_augmented_auroc": float(memory_augmented_auroc),
        "no_memory_baseline_auroc": float(no_memory_baseline_auroc),
        "n_examples_per_domain": n_train,
        "n_domains": 2,
        "sqlite_schema_version": memory.schema_version,
        "preconditions_checked": preconditions_checked,
        "duration_s": duration_s,
        "random_seed": seed
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_2512_fr11_tier2_memory.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Experiment JSON saved.")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
