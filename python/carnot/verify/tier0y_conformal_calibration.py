import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class ConformalCalibrationVerifier:
    def __init__(self, vectorizer, w, b):
        self.vectorizer = vectorizer
        self.w = w
        self.b = b

    def compute_energy(self, question: str, response: str) -> float:
        X = self.vectorizer.transform([response]).toarray()
        score = X @ self.w + self.b
        return float(score[0])
