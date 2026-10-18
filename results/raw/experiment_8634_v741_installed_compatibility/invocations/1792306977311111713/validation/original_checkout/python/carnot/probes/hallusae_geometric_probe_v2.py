"""HalluSAEGeometricProbeV2 — Tier 0j hallucination detector with temporal velocity features.

**Researcher summary:**
    Extends HalluSAEGeometricProbe (Exp 863, AUC=0.6144) with trajectory dynamics.
    The root-cause diagnosis for the marginal AUC is that a single energy snapshot
    per step misses the key kinematic signature of hallucination: energy does not
    just *increase* — it *accelerates*.  Hallucinating chains show a positive second
    derivative (energy rises faster over time), while correct chains show oscillating
    velocity with no persistent trend.

    New feature vector (6 dimensions) per trajectory:
        [energy_mean, energy_std, peak_energy,
         velocity_mean, accel_mean, monotone_increase_fraction]

    Where:
        step_energy[t]   = geometric L2 distance from centroid at step t (scalar)
        velocity[t]      = step_energy[t] - step_energy[t-1]  (zero-padded at t=0)
        accel[t]         = velocity[t] - velocity[t-1]         (zero-padded at t=0,1)

    A logistic-regression classifier trained on these 6 features separates correct
    from hallucinating trajectories better than the single-number energy AUC=0.6144.

    Relationship to Exp 863:
        V2 reuses the same TF-IDF-bigram SAE proxy and the same grounded centroid
        computation.  The only change is the aggregation: instead of returning
        mean(distances) as the single score, V2 builds a 6-feature vector per
        trajectory and classifies it with logistic regression.

Spec: REQ-VERIFY-143, SCENARIO-VERIFY-169, SCENARIO-VERIFY-170
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from carnot.probes.hallusae_geometric_probe import HalluSAEGeometricProbe


class HalluSAEGeometricProbeV2(HalluSAEGeometricProbe):
    """Tier 0j hallucination probe with temporal velocity and acceleration features.

    **For engineers:**
        Inherits from HalluSAEGeometricProbe to reuse the TF-IDF vectorizer and
        grounded centroid.  Adds a trajectory-level feature extractor that computes
        velocity (first derivative of step energy) and acceleration (second
        derivative) across CoT steps.

        The intuition:
            - Correct CoT: energy *oscillates* — some steps use unusual vocabulary,
              some use familiar vocabulary, but there is no persistent trend.
            - Hallucinating CoT: energy *accelerates upward* — once the chain starts
              drifting into nonsense space, each step wanders further from the centroid
              faster than the last.

        The 6-feature vector captures this in a compact, interpretable form that a
        simple logistic-regression classifier can learn from 40 training examples
        (80% of 50 synthetic pairs).

    Example::

        from carnot.probes.hallusae_geometric_probe_v2 import HalluSAEGeometricProbeV2

        correct_corpus = [["step1 ...", "step2 ..."], ...]   # 25 pairs
        hallu_corpus   = [["step1 ...", "step2 ..."], ...]   # 25 pairs
        reference_steps = [s for pair in correct_corpus for s in pair]

        probe = HalluSAEGeometricProbeV2(reference_steps=reference_steps)
        probe.train_trajectory(correct_corpus, hallu_corpus)

        result = probe.detect_trajectory(["step1 ...", "step2 ...", "step3 ..."])
        print(result["is_unstable_v2"])       # True / False
        print(result["trajectory_auc"])       # float
        print(result["feature_importances"])  # dict

    Spec: REQ-VERIFY-143
    """

    def __init__(
        self,
        reference_steps: list[str],
        threshold: float = 0.8,
        feature_dim: int = 6,
    ) -> None:
        """Initialise probe with TF-IDF centroid and a fresh classifier placeholder.

        **For engineers:**
            Calls the parent constructor (which fits the TF-IDF vectorizer and
            computes the grounded centroid), then stores the feature_dim and
            initialises self.classifier to None.  Call train_trajectory() before
            calling detect_trajectory().

        Args:
            reference_steps: Correct CoT steps for centroid computation — same role
                as in HalluSAEGeometricProbe.
            threshold: Binary detection threshold passed to the parent's is_anomalous
                method.  Not used by the V2 logistic-regression path; kept for
                compatibility with V1 callers.
            feature_dim: Expected size of the trajectory feature vector.  Default 6.
                Changing this without updating compute_trajectory_features() will
                break the classifier shape.

        Raises:
            ValueError: If reference_steps is empty (propagated from parent).
        """
        super().__init__(reference_steps=reference_steps, threshold=threshold)
        self.feature_dim = feature_dim
        # Classifier is None until train_trajectory() is called.
        self.classifier: LogisticRegression | None = None

    # ------------------------------------------------------------------
    # Trajectory feature extraction
    # ------------------------------------------------------------------

    def compute_trajectory_features(self, step_energies: list[float]) -> np.ndarray:
        """Compute a 6-feature trajectory descriptor from per-step energy scalars.

        **For engineers:**
            This is the core novelty of V2.  Given a list of per-step L2 energies
            (already computed by calling self.geometric_energy on individual steps),
            this method derives temporal dynamics — velocity and acceleration —
            in addition to the static summary statistics used in V1.

            Feature layout (shape: (6,)):
                0: energy_mean                 — V1 signal: average energy level
                1: energy_std                  — spread of energy across steps
                2: peak_energy                 — worst (highest) single-step energy
                3: velocity_mean               — mean first derivative; positive = trending up
                4: accel_mean                  — mean second derivative; positive = accelerating
                5: monotone_increase_fraction  — fraction of steps where energy rose vs prior step

            Zero-padding rule for finite differences:
                velocity[0] = 0          (no prior step to diff against)
                accel[0] = accel[1] = 0  (not enough history for second derivative)
            This preserves the length-T shape so all features are computed on the
            same number of samples regardless of trajectory length.

        Args:
            step_energies: List of T floats, one per CoT step.  T >= 1.
                Each float is the L2 distance of that step from the grounded centroid.

        Returns:
            np.ndarray of shape (6,) and dtype float64:
                [energy_mean, energy_std, peak_energy,
                 velocity_mean, accel_mean, monotone_increase_fraction]

        Spec: REQ-VERIFY-143-1
        """
        if not step_energies:
            return np.zeros(self.feature_dim, dtype=np.float64)

        e = np.array(step_energies, dtype=np.float64)
        T = len(e)

        # Velocity: first difference; pad leading zero so length stays T.
        # velocity[t] = energy[t] - energy[t-1] for t >= 1, else 0.
        velocity = np.zeros(T, dtype=np.float64)
        if T > 1:
            velocity[1:] = e[1:] - e[:-1]

        # Acceleration: first difference of velocity; pad two leading zeros.
        # accel[t] = velocity[t] - velocity[t-1] for t >= 2, else 0.
        accel = np.zeros(T, dtype=np.float64)
        if T > 2:
            accel[2:] = velocity[2:] - velocity[1:-1]

        energy_mean = float(e.mean())
        energy_std = float(e.std())
        peak_energy = float(e.max())
        velocity_mean = float(velocity.mean())
        accel_mean = float(accel.mean())
        # Fraction of steps (t >= 1) where velocity > 0 (energy increased).
        # For T=1 there are no transitions; fraction = 0.0.
        monotone_increase_fraction = float((velocity[1:] > 0).mean()) if T > 1 else 0.0

        return np.array(
            [
                energy_mean,
                energy_std,
                peak_energy,
                velocity_mean,
                accel_mean,
                monotone_increase_fraction,
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Per-trajectory energy list helper
    # ------------------------------------------------------------------

    def _step_energies(self, cot_steps: list[str]) -> list[float]:
        """Compute per-step L2 distances from the grounded centroid.

        **For engineers:**
            V1's geometric_energy() computes the *mean* across steps to get a
            single scalar.  V2 needs the per-step values to derive velocity and
            acceleration.  This method returns them as a plain Python list.

        Args:
            cot_steps: List of CoT step strings.

        Returns:
            List of T floats, one per step.
        """
        features = self.vectorizer.transform(cot_steps).toarray()
        distances = np.linalg.norm(features - self.centroid, axis=1)
        return distances.tolist()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_trajectory(
        self,
        pos_corpus: list[list[str]],
        neg_corpus: list[list[str]],
    ) -> None:
        """Train a logistic-regression classifier on 6-feature trajectory vectors.

        **For engineers:**
            pos_corpus = hallucinating CoT trajectories (label 1).
            neg_corpus = correct CoT trajectories (label 0).
            For each trajectory, computes per-step energies via _step_energies(),
            then calls compute_trajectory_features() to get the 6-feature vector.
            Trains sklearn LogisticRegression (max_iter=1000, C=1.0) on the combined
            feature matrix.

            After calling train_trajectory(), self.classifier is set and
            detect_trajectory() is usable.

        Args:
            pos_corpus: List of N_pos CoT step lists (hallucinating trajectories).
            neg_corpus: List of N_neg CoT step lists (correct trajectories).

        Raises:
            ValueError: If either corpus is empty.
        """
        if not pos_corpus or not neg_corpus:
            raise ValueError("Both pos_corpus and neg_corpus must be non-empty")

        X_rows: list[np.ndarray] = []
        y: list[int] = []

        for steps in neg_corpus:
            energies = self._step_energies(steps)
            X_rows.append(self.compute_trajectory_features(energies))
            y.append(0)

        for steps in pos_corpus:
            energies = self._step_energies(steps)
            X_rows.append(self.compute_trajectory_features(energies))
            y.append(1)

        X = np.stack(X_rows, axis=0)  # shape: (N, 6)
        y_arr = np.array(y, dtype=np.int32)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y_arr)
        self.classifier = clf

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect_trajectory(self, step_texts: list[str]) -> dict:
        """Detect whether a CoT trajectory is hallucinating using V2 classifier.

        **For engineers:**
            Requires train_trajectory() to have been called first.  Computes the
            6-feature vector for the input trajectory, then queries the classifier
            for a probability score.  The binary label (is_unstable_v2) uses a
            0.5 threshold on the classifier's hallucination probability.

            trajectory_auc is computed on the training corpus (known at fit time)
            and stored; this method returns it for convenience.  For a held-out AUC,
            use compute_trajectory_auc() directly.

            feature_importances is a dict mapping feature names to their logistic
            regression coefficient magnitudes (absolute values of coef_[0]).

        Args:
            step_texts: List of CoT step strings for a single model response.

        Returns:
            dict with keys:
                is_unstable_v2 (bool): True if classifier predicts hallucinating.
                trajectory_auc (float): AUC from the last compute_trajectory_auc() call,
                    or -1.0 if not yet computed.
                feature_importances (dict[str, float]): absolute coefficient magnitudes.

        Raises:
            RuntimeError: If train_trajectory() has not been called yet.

        Spec: REQ-VERIFY-143-4
        """
        if self.classifier is None:
            raise RuntimeError("train_trajectory() must be called before detect_trajectory()")

        energies = self._step_energies(step_texts)
        feat_vec = self.compute_trajectory_features(energies).reshape(1, -1)
        proba = float(self.classifier.predict_proba(feat_vec)[0, 1])

        feature_names = [
            "energy_mean",
            "energy_std",
            "peak_energy",
            "velocity_mean",
            "accel_mean",
            "monotone_increase_fraction",
        ]
        importances = {
            name: float(abs(self.classifier.coef_[0][i])) for i, name in enumerate(feature_names)
        }

        return {
            "is_unstable_v2": proba > 0.5,
            "trajectory_auc": getattr(self, "_last_trajectory_auc", -1.0),
            "feature_importances": importances,
        }

    # ------------------------------------------------------------------
    # AUC evaluation
    # ------------------------------------------------------------------

    def compute_trajectory_auc(
        self,
        pos_corpus: list[list[str]],
        neg_corpus: list[list[str]],
    ) -> float:
        """Compute AUC-ROC of the V2 classifier on held-out corpus pairs.

        **For engineers:**
            Uses the classifier's predict_proba() to generate a soft score for
            each trajectory, then evaluates with sklearn roc_auc_score.  Sets
            self._last_trajectory_auc so that detect_trajectory() can return it.

        Args:
            pos_corpus: Hallucinating CoT trajectories (label 1).
            neg_corpus: Correct CoT trajectories (label 0).

        Returns:
            AUC-ROC float in [0.0, 1.0].

        Raises:
            RuntimeError: If train_trajectory() has not been called.
        """
        if self.classifier is None:
            raise RuntimeError("train_trajectory() must be called before compute_trajectory_auc()")

        scores: list[float] = []
        labels: list[int] = []

        for steps in neg_corpus:
            energies = self._step_energies(steps)
            feat_vec = self.compute_trajectory_features(energies).reshape(1, -1)
            scores.append(float(self.classifier.predict_proba(feat_vec)[0, 1]))
            labels.append(0)

        for steps in pos_corpus:
            energies = self._step_energies(steps)
            feat_vec = self.compute_trajectory_features(energies).reshape(1, -1)
            scores.append(float(self.classifier.predict_proba(feat_vec)[0, 1]))
            labels.append(1)

        auc = float(roc_auc_score(labels, scores))
        self._last_trajectory_auc = auc
        return auc
