"""HalluSAEGeometricProbe — Tier 0i hallucination detector using SAE feature geometry.

**Researcher summary:**
    Implements a lightweight approximation of arXiv 2604.16430 (HalluSAE: Detecting
    Hallucinations via Sparse Autoencoder Feature Geometry).  The full HalluSAE method
    uses trained Sparse Autoencoders (SAEs) to project CoT steps into a feature space
    and measures how far each step drifts from a "grounded reasoning manifold" — the
    centroid of correct reasoning steps.  Energy rises when steps visit geometrically
    distant regions, which correlates with hallucination.

    The energy formula from the paper (adapted):
        E_geo(trajectory) = (1/T) * sum_t || f_SAE(step_t) - centroid_grounded ||_2

    **SAE proxy used here:**
    We do not have a trained SAE.  Instead we use TF-IDF bigrams (unigrams + bigrams,
    max 512 features) as a bag-of-features proxy for SAE activations.  This gives a
    geometry-aware feature space without requiring a trained neural network:
        - Vocabulary-based features capture lexical similarity with reference steps
        - Bigrams capture short phrase patterns that differ between correct/hallucinated steps
        - IDF weighting down-weights ubiquitous words, preserving meaningful signals

    This is explicitly labeled an approximation throughout — the AUC will be lower
    than a true SAE implementation because TF-IDF lacks the compositional structure
    of learned sparse representations.

**Tier 0i role in the cascade:**
    Advisory-only.  The probe computes geometric_energy and sets hallusae_anomalous
    on VerificationResult.  It does NOT block repair or change the verified flag.
    The caller integrates this probe the same way as Tier 0g (StreamingCoTHalluDetector):
    run it before or after the main cascade, write results into VerificationResult.

Spec: REQ-PROBE-050, SCENARIO-PROBE-060
"""

from __future__ import annotations

import numpy as np

# sklearn is already a dependency via the pipeline's constraint extraction path.
# TfidfVectorizer gives us a lightweight no-GPU feature extractor that approximates
# SAE activations via sparse token co-occurrence statistics.
from sklearn.feature_extraction.text import TfidfVectorizer


class HalluSAEGeometricProbe:
    """Tier 0i hallucination probe based on TF-IDF bigram geometry (SAE proxy).

    **For engineers:**
        Initialize with a list of correct/grounded reasoning steps.  The probe
        fits a TF-IDF vectorizer on those steps and stores their mean feature
        vector as the "grounded centroid" — the center of the correct reasoning
        manifold in this approximate feature space.

        At inference time, call geometric_energy(cot_steps) with the list of
        CoT steps from a model response.  The probe transforms each step into
        the same feature space, computes its L2 distance from the centroid, and
        returns the mean distance over all steps.  High mean distance → the
        trajectory visits unusual regions → potential hallucination.

        Call is_anomalous(cot_steps) to get a Boolean flag.  This is what the
        pipeline integration writes into VerificationResult.hallusae_anomalous.

    Example::

        from carnot.probes.hallusae_geometric_probe import HalluSAEGeometricProbe

        correct_steps = [
            "Let x = 5.  Then 2x = 10.",
            "Subtracting 3 gives 10 - 3 = 7.",
        ]
        probe = HalluSAEGeometricProbe(reference_steps=correct_steps)

        hallucinated_steps = [
            "Let x = 5.  Then 2x = 42 because magic.",
            "Therefore the answer is banana.",
        ]
        energy = probe.geometric_energy(hallucinated_steps)
        print(probe.is_anomalous(hallucinated_steps))  # True when energy > 0.8

    Spec: REQ-PROBE-050
    """

    def __init__(
        self,
        reference_steps: list[str],
        threshold: float = 0.8,
    ) -> None:
        """Fit the TF-IDF proxy and store the grounded centroid.

        **For engineers:**
            reference_steps are the "ground truth" CoT steps — examples of correct
            reasoning that define the grounded manifold.  More diverse examples
            improve centroid stability, but even a small set (10–50 steps) gives
            a usable signal.

            The vectorizer uses unigrams and bigrams (ngram_range=(1,2)) to capture
            both single-token and short-phrase patterns.  max_features=512 limits
            the vocabulary to keep memory and compute constant regardless of corpus
            size.  This mirrors the fixed-dimensionality property of SAE features.

            threshold controls is_anomalous().  The default of 0.8 was chosen to
            give roughly 70% specificity on 50-pair synthetic benchmarks; tune this
            on domain-specific reference sets for production use.

        Args:
            reference_steps: List of correct reasoning step strings.  Must be
                non-empty.  Used both to fit the vectorizer vocabulary and to
                compute the grounded centroid.
            threshold: Mean L2 distance above which a trajectory is flagged as
                anomalous.  Default 0.8.

        Raises:
            ValueError: If reference_steps is empty.
        """
        if not reference_steps:
            raise ValueError("reference_steps must be non-empty to fit the grounded centroid")

        self.threshold = threshold

        # Fit the TF-IDF vectorizer on the reference (grounded) steps.
        # ngram_range=(1,2): unigrams catch individual tokens, bigrams catch phrases.
        # max_features=512: fixed feature dimension regardless of reference corpus size —
        # mirrors how SAE feature dimensionality is fixed at training time.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=512)
        ref_features = self.vectorizer.fit_transform(reference_steps).toarray()

        # Centroid = mean of all reference feature vectors.
        # This is the "grounded manifold center" in TF-IDF space.
        # Shape: (max_features,)  — a single representative point.
        self.centroid: np.ndarray = ref_features.mean(axis=0)

    def geometric_energy(self, cot_steps: list[str]) -> float:
        """Compute mean L2 distance of CoT steps from the grounded centroid.

        **For engineers:**
            This is the core energy signal.  Each step is projected into TF-IDF
            feature space (using the vocabulary fitted on reference steps), then
            its Euclidean distance from the centroid is computed.  The mean over
            all steps is the trajectory-level geometric energy.

            A step that uses vocabulary and bigrams similar to the reference set
            will project close to the centroid → low distance → low energy.
            A step with unusual vocabulary (injected nonsense, factual errors with
            different word choices) will project far from the centroid → high energy.

            Empty or out-of-vocabulary steps will produce a zero vector, which has
            distance equal to ||centroid||.  This is conservative: such steps are
            scored as "as far from grounded as a random point would be."

        Args:
            cot_steps: List of CoT step strings for a single model response.
                Must be non-empty.

        Returns:
            Float: mean L2 distance from the grounded centroid.  Range [0, ∞).
            Lower = more similar to reference reasoning.  Higher = more anomalous.

        Spec: REQ-PROBE-050
        """
        features = self.vectorizer.transform(cot_steps).toarray()
        # distances shape: (n_steps,)
        distances = np.linalg.norm(features - self.centroid, axis=1)
        return float(distances.mean())

    def is_anomalous(self, cot_steps: list[str]) -> bool:
        """Return True when geometric_energy exceeds the detection threshold.

        **For engineers:**
            This is the Boolean flag that the pipeline writes into
            VerificationResult.hallusae_anomalous.  Advisory-only: a True result
            does not alter the verified flag or trigger repair by itself.

            The threshold should be calibrated on a domain-representative reference
            set.  The default of 0.8 was chosen for the synthetic 50-pair benchmark
            in Exp 863.

        Args:
            cot_steps: List of CoT step strings for a single model response.

        Returns:
            True if mean L2 distance from centroid > threshold, False otherwise.

        Spec: REQ-PROBE-050
        """
        return self.geometric_energy(cot_steps) > self.threshold
