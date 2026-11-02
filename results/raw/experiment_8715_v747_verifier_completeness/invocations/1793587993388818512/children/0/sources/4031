"""Tier-4 Adaptive Structure Prototype."""

from typing import List, Dict, Tuple, Callable
import numpy as np

def compute_marginal_contributions(
    labels: np.ndarray,
    scores_by_verifier: Dict[str, np.ndarray],
    weights: Dict[str, float],
    auroc_fn: Callable[[np.ndarray, np.ndarray], float],
) -> Dict[str, float]:
    """Compute the marginal contribution of each verifier."""
    contributions = {}
    active_verifiers = list(scores_by_verifier.keys())
    
    # Calculate full ensemble score
    ensemble_scores = sum(weights.get(v, 0.0) * scores_by_verifier[v] for v in active_verifiers)
    full_auroc = auroc_fn(labels, ensemble_scores)
    
    for v in active_verifiers:
        dropped_scores = sum(
            weights.get(name, 0.0) * scores_by_verifier[name]
            for name in active_verifiers if name != v
        )
        dropped_auroc = auroc_fn(labels, dropped_scores)
        contributions[v] = full_auroc - dropped_auroc
        
    return contributions

def prune_verifiers(
    marginal_contributions: Dict[str, float],
    threshold: float = 0.002
) -> Tuple[List[str], List[str]]:
    """Prune verifiers below threshold. Returns (pruned, retained)."""
    pruned = []
    retained = []
    for v, m in marginal_contributions.items():
        if m < threshold:
            pruned.append(v)
        else:
            retained.append(v)
    return pruned, retained

def flag_residual_regions(
    labels: np.ndarray,
    scores_by_verifier: Dict[str, np.ndarray],
    retained_verifiers: List[str],
    threshold: float = 0.5
) -> List[int]:
    """Flag gold-incorrect candidates missed by all retained verifiers."""
    residual = []
    for i, label in enumerate(labels):
        if label == 1:
            catches = [scores_by_verifier[v][i] >= threshold for v in retained_verifiers]
            if not any(catches):
                residual.append(int(i))
    return residual
