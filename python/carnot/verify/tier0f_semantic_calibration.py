"""Tier 0f: Semantic calibrated verifier."""

import os
import pickle

class SemanticCalibratedVerifier:
    """TF-IDF logistic verifier with paraphrase clustering calibration."""

    def __init__(self, model_path: str = "results/tier0f_semantic_calibrated_logistic.pkl"):
        self.model_path = model_path
        self.vectorizer = None
        self.clf = None
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.clf = data["clf"]

    def verify(self, text: str) -> float:
        """Verify text and return probability of being correct."""
        if self.clf is None or self.vectorizer is None:
            return 0.5
        X = self.vectorizer.transform([text])
        return float(self.clf.predict_proba(X)[0, 1])
