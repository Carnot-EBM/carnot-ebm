"""Tier 0e: TF-IDF margin-ranking logistic model.

The persisted model is a pickle, not safetensors, because it holds fitted
scikit-learn estimators (a TF-IDF vectoriser and a logistic classifier) --
Python object graphs, not a tensor dict, so safetensors cannot represent them.
Pickle deserialization executes arbitrary code, so the load is routed through
``carnot.serialization_safety.safe_pickle_load``, which refuses any path
resolving outside this checkout.  See that module for what the restriction does
and does not buy.
"""

import os

from carnot.serialization_safety import safe_pickle_load


class EORMVerifier:
    """TF-IDF margin-ranking logistic verifier trained on FoVer."""

    def __init__(self, model_path: str = "results/tier0e_model.pkl"):
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
