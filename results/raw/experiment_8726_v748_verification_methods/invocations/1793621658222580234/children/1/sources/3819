import numpy as np
from sklearn.linear_model import LogisticRegression

class FEPAggregator:
    """
    Aggregates Free Energy Principle (FEP) signals from multiple verifiers.
    
    Uses Strategy 2: Learned Logistic Regression.
    This naturally handles varying scales and signs (anti-correlated verifiers) 
    that break a simple sum factor graph.
    """
    def __init__(self, coefficients=None, intercept=0.0):
        self.coefficients = np.array(coefficients) if coefficients is not None else None
        self.intercept = intercept
        self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        if self.coefficients is not None:
            self.model.coef_ = np.array([self.coefficients])
            self.model.intercept_ = np.array([self.intercept])
            self.model.classes_ = np.array([0, 1])
        
    def fit(self, feature_matrix, labels):
        """
        Fit the logistic regression model on calibration data.
        
        Args:
            feature_matrix: Array of shape (n_samples, n_verifiers)
            labels: Array of shape (n_samples,) where 1=incorrect, 0=correct
        """
        self.model.fit(feature_matrix, labels)
        self.coefficients = self.model.coef_[0]
        self.intercept = self.model.intercept_[0]
        
    def aggregate(self, energies):
        """
        Aggregate a single row or multiple rows of energies into a joint FEP score.
        High score means high energy (incorrect).
        
        Args:
            energies: List or Array of verifier energies.
        Returns:
            Joint FEP score (probability of being incorrect).
        """
        x = np.array(energies)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        if self.coefficients is not None:
            z = np.dot(x, self.coefficients) + self.intercept
            prob = 1.0 / (1.0 + np.exp(-z))
            return float(prob[0]) if prob.shape[0] == 1 else prob
        else:
            raise ValueError("Aggregator not fitted and no coefficients provided.")
