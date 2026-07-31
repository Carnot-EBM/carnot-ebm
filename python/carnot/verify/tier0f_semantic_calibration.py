"""Tier 0f: Semantic calibrated verifier.

Persisted as a pickle rather than safetensors for the same reason as Tier 0e:
the payload is fitted scikit-learn estimators, not tensors.  The load goes
through ``carnot.serialization_safety.safe_pickle_load``, which confines
code-executing deserialization to in-repo artifacts.
"""

import os

from carnot.serialization_safety import safe_pickle_load


class SemanticCalibratedVerifier:
    """TF-IDF logistic verifier with paraphrase clustering calibration."""

    def __init__(self, model_path: str = "results/tier0f_semantic_calibrated_logistic.pkl"):
        self.model_path = model_path
        self.vectorizer = None
        self.clf = None
        if os.path.exists(self.model_path):
            data = safe_pickle_load(self.model_path, expected_type=dict)
            self.vectorizer = data["vectorizer"]
            self.clf = data["clf"]

    def verify(self, text: str) -> float:
        """Verify text and return probability of being correct."""
        if self.clf is None or self.vectorizer is None:
            return 0.5
        X = self.vectorizer.transform([text])
        return float(self.clf.predict_proba(X)[0, 1])
