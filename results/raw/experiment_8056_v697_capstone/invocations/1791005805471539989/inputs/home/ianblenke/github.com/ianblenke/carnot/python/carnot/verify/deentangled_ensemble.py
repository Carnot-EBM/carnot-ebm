import numpy as np

class DeentangledEnsemble:
    """
    De-entangled Ensemble reweights verifier components based on their
    pairwise behavioral entanglement, assigning lower weight to highly
    correlated (entangled) components, as per arXiv:2604.07650.
    """
    def __init__(self):
        self.weights_ = None
        
    def fit(self, verifier_scores_matrix: np.ndarray, labels: np.ndarray):
        """
        Compute de-entangled weights from the Pearson correlation matrix
        of the verifier scores.
        
        Args:
            verifier_scores_matrix: shape (n_samples, n_components)
            labels: shape (n_samples,) - not strictly used for correlation but 
                    provided for API consistency and potential future use.
        Returns:
            self
        """
        n_samples, n_components = verifier_scores_matrix.shape
        if n_components < 2:
            self.weights_ = np.ones(n_components)
            return self
            
        corr_matrix = np.corrcoef(verifier_scores_matrix, rowvar=False)
        weights = np.zeros(n_components)
        
        for i in range(n_components):
            # sum of |corr(i, j)| for j != i
            sum_abs_corr = np.sum(np.abs(corr_matrix[i, :])) - 1.0
            weights[i] = 1.0 - (sum_abs_corr / (n_components - 1))
            
        # Normalize weights to sum to 1
        sum_weights = np.sum(weights)
        if sum_weights != 0:
            self.weights_ = weights / sum_weights
        else:
            self.weights_ = np.ones(n_components) / n_components
            
        return self
        
    def predict(self, verifier_scores_matrix: np.ndarray) -> np.ndarray:
        """
        Apply computed weights to the verifier scores matrix.
        
        Args:
            verifier_scores_matrix: shape (n_samples, n_components)
        Returns:
            weighted_scores: shape (n_samples,)
        """
        if self.weights_ is None:
            raise ValueError("DeentangledEnsemble is not fitted.")
        return verifier_scores_matrix @ self.weights_
