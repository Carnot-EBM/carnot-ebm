"""Proximal-Gradient step for constraint satisfaction in the verifier pipeline."""

from typing import List, Optional
import numpy as np

def continuous_relaxation_penalty(logits: np.ndarray, constraints: List[callable]) -> float:
    """Compute soft penalty using continuous relaxation of constraints."""
    penalty = 0.0
    for constraint in constraints:
        penalty += constraint(logits)
    return penalty

def proximal_descent_projection(logits: np.ndarray, constraints: List[callable], step_size: float = 0.1, num_steps: int = 10) -> np.ndarray:
    """Implement proximal-descent projection over logits."""
    current_logits = logits.copy()
    for _ in range(num_steps):
        # Compute numerical gradient of the penalty
        grad = np.zeros_like(current_logits)
        eps = 1e-4
        for i in range(len(current_logits)):
            logits_plus = current_logits.copy()
            logits_plus[i] += eps
            penalty_plus = continuous_relaxation_penalty(logits_plus, constraints)
            
            logits_minus = current_logits.copy()
            logits_minus[i] -= eps
            penalty_minus = continuous_relaxation_penalty(logits_minus, constraints)
            
            grad[i] = (penalty_plus - penalty_minus) / (2 * eps)
            
        current_logits = current_logits - step_size * grad
    return current_logits

def measure_constraint_satisfaction_improvement(original_logits: np.ndarray, projected_logits: np.ndarray, constraints: List[callable]) -> float:
    """Measure constraint satisfaction improvement versus soft penalty."""
    orig_penalty = continuous_relaxation_penalty(original_logits, constraints)
    proj_penalty = continuous_relaxation_penalty(projected_logits, constraints)
    return orig_penalty - proj_penalty
