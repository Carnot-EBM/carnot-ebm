"""IAS (Instance-Adaptive Scaling) gate calibration for EnsembleGate.

**Why this module exists:**
    EnsembleGate v3 and v4 use a fixed threshold=0.30 applied identically to all
    three extractors (symcode, structured, causal).  This fails when different
    extractors have different variance distributions: a recall of 0.36 from the
    causal extractor is meaningfully different from 0.36 from the symcode extractor
    because the causal extractor's scores fluctuate more.  A single fixed threshold
    ignores this heterogeneity.

    arXiv 2506.09338 (IAS: Instance-Adaptive Scaling) demonstrates that quantile
    regression calibration aligns confidence estimates with true success probabilities
    by learning adaptive thresholds that respect per-extractor variance.  We apply
    the 10th-percentile rule: the gate threshold for each extractor is set to the
    10th percentile of that extractor's recall distribution over FOVER pairs.
    High-variance extractors naturally have a lower 10th-percentile, so they get
    a more permissive threshold; low-variance extractors get a tighter one.

**Key design choices:**
    - Pinball loss minimisation is used to fit the quantile because it is the
      canonical loss function for quantile regression and has a closed-form solution
      for the unconditional case: the q-th quantile of the observed distribution.
    - We implement a simple sorted-percentile estimator (no JAX gradient descent
      needed for the unconditional case) to keep the module fast and dependency-free.
    - The FOVER pairs file supplies recall signals via the 'label' and 'confidence'
      fields: a step labeled 'correct' with confidence C contributes recall C;
      a step labeled 'incorrect' contributes recall 0.0 for the verification step.

**Extractor definitions (how per-extractor recall is derived from FOVER pairs):**
    - symcode: arithmetic correctness — 'correct' label implies arithmetic is verifiable.
    - structured: presence of COMPUTE: format — all labeled steps are considered
      structured (extractor fires on every step regardless of label).
    - causal: causal reasoning step — 'correct' label implies causal chain is sound.

Spec: REQ-VERIFY-151, REQ-VERIFY-152, SCENARIO-VERIFY-200, SCENARIO-VERIFY-201
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# QuantileRegressionHead
# ---------------------------------------------------------------------------


class QuantileRegressionHead:
    """Simple per-extractor quantile regression using pinball loss.

    For the unconditional case (single distribution, no features), the pinball
    loss minimiser at quantile q equals the empirical q-th quantile of the data.
    This is a well-known closed-form result: minimising

        L_q(y, y_hat) = q*(y - y_hat)   if y > y_hat
                        (q-1)*(y - y_hat) if y <= y_hat

    over a constant y_hat gives y_hat = quantile(observations, q).

    We therefore implement train() as a sorted-percentile lookup — algebraically
    identical to gradient descent on pinball loss at convergence, but O(n log n)
    instead of iterative.

    Spec: REQ-VERIFY-151-1
    """

    def train(self, observations: List[float], quantile: float = 0.10) -> float:
        """Fit pinball loss at the given quantile and return the threshold.

        The pinball (check) loss for quantile q has its global minimum at the
        empirical q-th percentile of the observation distribution.  For the
        unconditional (intercept-only) case this reduces to a sorted lookup,
        which is what we compute here.

        Args:
            observations: List of recall values in [0, 1] from FOVER pair scoring.
                Each element represents one extractor's recall on one FOVER pair.
            quantile: The quantile to fit, in (0, 1).  Default is 0.10 (10th
                percentile) per REQ-VERIFY-152.

        Returns:
            The q-th quantile of the observation distribution — i.e. the adaptive
            gate threshold for this extractor.

        Raises:
            ValueError: If observations is empty or quantile is outside (0, 1).

        Spec: REQ-VERIFY-151-1, REQ-VERIFY-152-1
        """
        if not observations:
            raise ValueError("observations must be non-empty")
        if not (0.0 < quantile < 1.0):
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")

        sorted_obs = sorted(observations)
        n = len(sorted_obs)
        # Linear interpolation between adjacent ranks (standard percentile method).
        # Index: (n-1) * q gives the fractional rank position.
        idx = (n - 1) * quantile
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_obs[lo] * (1.0 - frac) + sorted_obs[hi] * frac


# ---------------------------------------------------------------------------
# IASGateCalibration
# ---------------------------------------------------------------------------


@dataclass
class IASGateCalibration:
    """Calibrated per-extractor gate thresholds from IAS quantile regression.

    Each threshold is the 10th percentile of that extractor's recall distribution
    over FOVER pairs.  High-variance extractors have a lower 10th-percentile
    (more permissive threshold); low-variance extractors have a higher one.

    Fields:
        symcode_threshold: Gate threshold for the SymCode arithmetic verifier.
            Set to the 10th percentile of symcode recall over FOVER pairs.
        structured_threshold: Gate threshold for the StructuredEquationForcer.
            Set to the 10th percentile of structured recall over FOVER pairs.
        causal_threshold: Gate threshold for the CausalReasoningVerifier.
            Set to the 10th percentile of causal recall over FOVER pairs.
        calibrated_from_n: Number of FOVER pairs used to fit the calibration.

    Spec: REQ-VERIFY-151-3
    """

    symcode_threshold: float
    structured_threshold: float
    causal_threshold: float
    calibrated_from_n: int


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def calibrate(fover_pairs_path: str) -> IASGateCalibration:
    """Calibrate per-extractor gate thresholds from labeled FOVER pairs.

    Loads the FOVER pairs JSON, derives per-extractor recall distributions, then
    fits a QuantileRegressionHead at q=0.10 for each extractor.

    **How extractor recall is derived from FOVER pairs:**
        Each FOVER pair has a 'label' ('correct'/'incorrect') and a 'confidence'
        (float in [0, 1]) representing the human annotator's certainty.

        - symcode recall per pair: confidence if label=='correct', else 0.0.
          Rationale: arithmetic correctness (symcode's domain) maps directly to
          the correctness label.
        - structured recall per pair: confidence (always).
          Rationale: the StructuredEquationForcer checks format presence, not
          correctness — any labeled step has a COMPUTE: format signal.
        - causal recall per pair: confidence if label=='correct', else 0.0.
          Rationale: causal chain soundness maps to the correctness label in the
          same way as symcode.

    Args:
        fover_pairs_path: Path to JSON file containing a list of FOVER pair dicts,
            each with keys 'question_id', 'step_text', 'label', 'confidence'.

    Returns:
        IASGateCalibration with per-extractor 10th-percentile thresholds.

    Raises:
        FileNotFoundError: If fover_pairs_path does not exist.
        KeyError: If FOVER pairs are missing required fields.

    Spec: REQ-VERIFY-151-2, SCENARIO-VERIFY-200
    """
    with open(fover_pairs_path) as f:
        pairs = json.load(f)

    symcode_recalls: List[float] = []
    structured_recalls: List[float] = []
    causal_recalls: List[float] = []

    for pair in pairs:
        label = pair["label"]
        confidence = float(pair["confidence"])
        recall_value = confidence if label == "correct" else 0.0

        # Symcode: arithmetic correctness — mirrors correctness label.
        symcode_recalls.append(recall_value)
        # Structured: format presence — fires on every step regardless of label.
        structured_recalls.append(confidence)
        # Causal: causal chain soundness — mirrors correctness label.
        causal_recalls.append(recall_value)

    head = QuantileRegressionHead()
    return IASGateCalibration(
        symcode_threshold=head.train(symcode_recalls, quantile=0.10),
        structured_threshold=head.train(structured_recalls, quantile=0.10),
        causal_threshold=head.train(causal_recalls, quantile=0.10),
        calibrated_from_n=len(pairs),
    )


# ---------------------------------------------------------------------------
# adaptive_gate_open
# ---------------------------------------------------------------------------


def adaptive_gate_open(
    calibration: IASGateCalibration,
    symcode: float,
    structured: float,
    causal: float,
) -> bool:
    """Return True when any extractor exceeds its calibrated IAS threshold.

    The gate opens when at least one extractor's observed recall meets or exceeds
    that extractor's calibrated 10th-percentile threshold.  This OR-logic means
    a single strong extractor can authorise VR — the same philosophy as
    EnsembleGate v4, but with adaptive per-extractor thresholds instead of
    fixed global ones.

    Args:
        calibration: IASGateCalibration from calibrate(), holding per-extractor
            10th-percentile thresholds.
        symcode: Observed SymCode extractor recall for this request.
        structured: Observed StructuredEquationForcer recall for this request.
        causal: Observed CausalReasoningVerifier recall for this request.

    Returns:
        True if gate opens (VR authorised), False if gate closes (VR blocked).

    Spec: REQ-VERIFY-152-3, SCENARIO-VERIFY-201
    """
    return (
        symcode >= calibration.symcode_threshold
        or structured >= calibration.structured_threshold
        or causal >= calibration.causal_threshold
    )
