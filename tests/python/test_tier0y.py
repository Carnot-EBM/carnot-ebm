import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from carnot.verify.tier0y_conformal_calibration import ConformalCalibrationVerifier

def test_conformal_calibration_verifier():
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(["dummy text"]).toarray()
    w = np.zeros(X_train.shape[1])
    b = 0.1
    verifier = ConformalCalibrationVerifier(vectorizer, w, b)
    score = verifier.compute_energy("question?", "dummy text")
    assert isinstance(score, float)
