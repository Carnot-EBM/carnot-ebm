"""Ensemble reweighting algorithms.

Implements the de-entangled reweighting algorithm for k=16 ensembles
from arXiv:2604.07650.

Spec: REQ-VERIFY-1732
"""

import numpy as np


class DeentangledReweighter:
    """Adjusts verifier weights based on failure covariance.
    
    Addresses correlated failures in AND-composition by de-entangling
    the weights using the inverse covariance matrix of failures.
    """

    def __init__(self, ridge: float = 1e-4) -> None:
        self.ridge = ridge
        self.weights_: np.ndarray | None = None

    def fit(self, failure_matrix: np.ndarray) -> "DeentangledReweighter":
        """Compute optimal weights from a boolean/float failure matrix.
        
        Args:
            failure_matrix: (N_samples, k_verifiers) where 1.0 means the verifier failed.
            
        Returns:
            self
        """
        cov = np.cov(failure_matrix, rowvar=False)
        k = cov.shape[0]
        # Add ridge for numerical stability
        cov += np.eye(k) * self.ridge
        
        # Calculate weights W = Sigma^{-1} 1
        inv_cov = np.linalg.inv(cov)
        ones = np.ones(k)
        w = inv_cov @ ones
        
        # Normalize weights
        w = np.clip(w, 0, None)  # Ensure non-negative weights
        w_sum = w.sum()
        if w_sum > 0:
            self.weights_ = w / w_sum
        else:
            self.weights_ = np.ones(k) / k
            
        return self

    def predict_weighted_score(self, failure_matrix: np.ndarray) -> np.ndarray:
        """Calculate weighted failure scores.
        
        Args:
            failure_matrix: (N_samples, k_verifiers)
            
        Returns:
            (N_samples,) weighted failure scores
        """
        if self.weights_ is None:
            raise ValueError("Reweighter is not fitted.")
        return failure_matrix @ self.weights_
