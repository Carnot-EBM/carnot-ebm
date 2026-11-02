import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def diversity_select(verifier_scores_matrix: np.ndarray, labels: np.ndarray, k_target: int = 4) -> list:
    """Greedy selection: start with best single verifier (highest AUROC), then add each
    verifier that maximizes F1 improvement on the examples the current ensemble misses.
    This targets recall gaps (examples no current verifier catches) rather than
    correlation reduction (entanglement premise)."""
    n_samples, n_verifiers = verifier_scores_matrix.shape
    labels = np.array(labels)
    
    # 1. Best single verifier
    best_auroc = -1
    best_idx = -1
    for i in range(n_verifiers):
        try:
            auc = roc_auc_score(labels, verifier_scores_matrix[:, i])
            if auc > best_auroc:
                best_auroc = auc
                best_idx = i
        except ValueError:
            pass
            
    if best_idx == -1:
        best_idx = 0  # Fallback
    
    selected = [best_idx]
    
    # 2. Greedily add to maximize F1 on missed examples
    while len(selected) < k_target and len(selected) < n_verifiers:
        current_scores = np.mean(verifier_scores_matrix[:, selected], axis=1)
        current_preds = (current_scores > 0.5).astype(int)
        
        missed_mask = (current_preds != labels)
        
        if not np.any(missed_mask):
            remaining = [i for i in range(n_verifiers) if i not in selected]
            if remaining:
                selected.append(remaining[0])
            break
            
        best_f1 = -1
        best_candidate = -1
        
        # Compute F1 of each remaining verifier ONLY on the missed examples
        for i in range(n_verifiers):
            if i in selected:
                continue
                
            candidate_preds = (verifier_scores_matrix[:, i] > 0.5).astype(int)
            missed_labels = labels[missed_mask]
            missed_preds = candidate_preds[missed_mask]
            
            try:
                # Calculate F1 on the missed subset
                if len(np.unique(missed_labels)) > 1:
                    f1 = f1_score(missed_labels, missed_preds, zero_division=0)
                else:
                    # Fallback if only one class is present in the missed set
                    f1 = np.mean(missed_labels == missed_preds)
            except ValueError:
                f1 = 0
                
            if f1 > best_f1:
                best_f1 = f1
                best_candidate = i
                
        if best_candidate != -1:
            selected.append(best_candidate)
        else:
            remaining = [i for i in range(n_verifiers) if i not in selected]
            if remaining:
                selected.append(remaining[0])
            else:
                break
                
    return selected
