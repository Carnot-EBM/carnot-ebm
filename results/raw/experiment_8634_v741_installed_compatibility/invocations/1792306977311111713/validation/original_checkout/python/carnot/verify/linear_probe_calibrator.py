"""Linear probe calibrator for Tier 0e features."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

class LinearProbeCalibrator:
    def __init__(self):
        self.calibrated_clf = None

    def fit(self, features, labels):
        # Base logistic regression with uncalibrated probabilities
        base_clf = LogisticRegression(random_state=42)
        # Use Isotonic regression with 5-fold CV to calibrate the probabilities
        self.calibrated_clf = CalibratedClassifierCV(estimator=base_clf, cv=5, method='isotonic')
        self.calibrated_clf.fit(features, labels)
        return self

    def calibrate(self, features):
        if self.calibrated_clf is None:
            raise ValueError("Calibrator is not fitted yet.")
        return self.calibrated_clf.predict_proba(features)[:, 1]

    def ece(self, predictions, labels, n_bins=10):
        predictions = np.array(predictions)
        labels = np.array(labels)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        
        # Adjust upper bound slightly to ensure inclusion of 1.0
        bin_uppers = bin_boundaries[1:]
        bin_uppers[-1] += 1e-6
        
        ece_val = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                ece_val += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return float(ece_val)
