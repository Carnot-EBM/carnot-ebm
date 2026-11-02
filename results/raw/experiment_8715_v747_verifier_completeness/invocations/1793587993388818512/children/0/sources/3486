import numpy as np
from scipy.stats import skew
from sklearn.neural_network import MLPClassifier

class JEPAViolationPredictor:
    def __init__(self, random_state=42, max_iter=200):
        self.predictor = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            random_state=random_state,
            max_iter=max_iter,
            activation='relu'
        )
        
    def extract_features(self, logprobs):
        """
        Extract features from the first half of the response logprobs.
        """
        if not logprobs:
            return [0.0, 0.0, 0.0, 0.0, 0]
            
        # First half of response
        partial_len = len(logprobs) // 2
        
        # Ensure we have at least one token
        if partial_len == 0:
            partial_len = 1
            
        partial_logprobs = logprobs[:partial_len]
        
        features = [
            float(np.mean(partial_logprobs)),
            float(np.var(partial_logprobs)),
            float(np.min(partial_logprobs)),
            float(skew(partial_logprobs) if len(partial_logprobs) > 1 else 0.0),
            float(len(partial_logprobs))
        ]
        
        # Replace NaN with 0.0 just in case
        return [0.0 if np.isnan(f) else f for f in features]
        
    def fit(self, X, y):
        self.predictor.fit(X, y)
        
    def predict_proba(self, X):
        return self.predictor.predict_proba(X)
        
    def get_feature_names(self):
        return ['mean_logprob', 'var_logprob', 'min_logprob', 'skew_logprob', 'partial_length']
