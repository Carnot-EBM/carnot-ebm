import numpy as np

class MonitorProvenanceAxis:
    """
    Implements a cached, batchable verifier axis that improves diversity measurement
    without reviving retired serial monitoring.
    """
    def __init__(self, axis_name: str = "trajectory_consistency"):
        self.axis_name = axis_name

    def evaluate(self, cached_candidates: list[dict]) -> np.ndarray:
        """
        Evaluate the axis over a batch of cached candidates.
        Returns a numpy array of scores (e.g., 0.0 to 1.0).
        """
        scores = []
        for cand in cached_candidates:
            if "trajectory_steps" in cand and cand["trajectory_steps"]:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return np.array(scores, dtype=float)

    def compute_max_correlation(self, axis_scores: np.ndarray, existing_columns: dict[str, np.ndarray]) -> float:
        """
        Compute the max correlation against existing columns.
        """
        max_corr = 0.0
        if not existing_columns or len(axis_scores) == 0:
            return 0.0
        for name, col in existing_columns.items():
            if len(col) != len(axis_scores):
                continue
            std_axis = np.std(axis_scores)
            std_col = np.std(col)
            if std_axis == 0 or std_col == 0:
                continue
            corr = np.abs(np.corrcoef(axis_scores, col)[0, 1])
            if corr > max_corr:
                max_corr = corr
        return float(max_corr)
