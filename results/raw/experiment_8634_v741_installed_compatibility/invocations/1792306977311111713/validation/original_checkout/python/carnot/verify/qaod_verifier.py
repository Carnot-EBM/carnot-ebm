"""QAOD Verifier implementation."""
import numpy as np

class QAODProbe:
    """QAOD-class probe that uses orthogonal decomposition of answer representations against question context.
    
    Per arXiv:2605.14449, project each answer representation onto the orthogonal complement
    of the question context subspace. The residual magnitude is the hallucination indicator.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, features: np.ndarray, labels: np.ndarray = None):
        """No-op for QAOD as it's a white-box geometric projection, not a learned probe (except maybe threshold)."""
        pass

    def compute_residual_magnitude(self, answers: np.ndarray, questions: np.ndarray) -> np.ndarray:
        """Compute the magnitude of the answer projected onto the orthogonal complement of the question.
        
        Args:
            answers: shape (N, D)
            questions: shape (N, D)
        Returns:
            residuals: shape (N,)
        """
        residuals = []
        for a, q in zip(answers, questions):
            q_norm_sq = np.dot(q, q)
            if q_norm_sq > 1e-12:
                proj = (np.dot(a, q) / q_norm_sq) * q
            else:
                proj = np.zeros_like(q)
            a_perp = a - proj
            residuals.append(np.linalg.norm(a_perp))
        return np.array(residuals)

    def predict_proba(self, answers: np.ndarray, questions: np.ndarray) -> np.ndarray:
        """Returns pseudo-probabilities for hallucination detection.
        Since we want TPR/FPR, the residual magnitude itself can be used as a continuous score for thresholding.
        In this implementation we return the residual magnitude as the score (higher means more hallucination).
        Wait, for NLA, the predict_proba returns [prob_class_0, prob_class_1].
        We'll just return a 2D array [1-score, score] assuming score is normalized between 0 and 1.
        """
        residuals = self.compute_residual_magnitude(answers, questions)
        # Normalize to 0-1 for probability if needed, but for thresholding any continuous score works.
        # Let's use a logistic function to map to 0-1.
        probs = 1.0 / (1.0 + np.exp(- (residuals - self.threshold)))
        return np.vstack([1 - probs, probs]).T
        
    def predict(self, answers: np.ndarray, questions: np.ndarray) -> np.ndarray:
        """Predict binary label (1 for hallucination/disagreement, 0 for agreement)."""
        residuals = self.compute_residual_magnitude(answers, questions)
        return (residuals > self.threshold).astype(int)
