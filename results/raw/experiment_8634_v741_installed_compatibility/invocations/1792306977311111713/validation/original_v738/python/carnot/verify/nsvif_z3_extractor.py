from carnot.extraction.nsvif_extractor import NsvifExtractor as NSVIFExtractor
import math

class SoundnessCompletenessTracker:
    def __init__(self, n_features: int = 1000):
        self.soundness_mistakes = 0
        self.completeness_mistakes = 0
        self.n_total = 0
        self.n_features = n_features

    def update(self, prediction: bool, label: bool):
        """
        prediction: True if safe, False if violation
        label: True if safe, False if violation
        """
        self.n_total += 1
        if prediction and not label:
            # prediction=safe, label=violation
            self.soundness_mistakes += 1
        elif not prediction and label:
            # prediction=violation, label=safe
            self.completeness_mistakes += 1

    def soundness_rate(self) -> float:
        return self.soundness_mistakes / self.n_total if self.n_total > 0 else 0.0

    def completeness_rate(self) -> float:
        return self.completeness_mistakes / self.n_total if self.n_total > 0 else 0.0

    def littlestone_soundness_bound(self) -> float:
        # sqrt(2 * soundness_mistakes * log(n_features))
        if self.n_features <= 0 or self.soundness_mistakes < 0:
            return 0.0
        return math.sqrt(2 * self.soundness_mistakes * math.log(self.n_features))

