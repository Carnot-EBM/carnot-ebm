"""Spilled Energy hallucination detector — no constraint extraction required.

**Researcher summary:**
    Implements the Spilled Energy signal from ICLR 2026 (arXiv 2602.18671) and
    the AR-EBM lookahead energy from arXiv 2512.15605.  Both signals are
    computable directly from raw LLM logit arrays, bypassing the brittle
    constraint-extraction bottleneck that limited Experiments 279 and earlier.

**Detailed explanation for engineers:**
    The fundamental insight: LLMs already compute an implicit energy over their
    outputs.  We don't need to *extract* constraints from the text — we can read
    the hallucination signal directly from the logit distribution.

    Two measures are provided:

    1. **Spilled Energy** (per token, then aggregated):
         spill_t = H(softmax(logit_t)) − max(log_softmax(logit_t))

       Interpretation: entropy tells us how "spread out" the model's uncertainty
       is.  The max log-prob term is the log-probability of the most likely token.
       When the model is confident (peaked distribution) both are small and nearly
       cancel, giving low spill.  When the model is confused (flat distribution)
       entropy is high and the max log-prob is small in magnitude, so spill is
       large.  High mean spill → the model was hedging → suspected hallucination.

    2. **Lookahead Energy** (AR-EBM bijection, arXiv 2512.15605):
         lookahead = −mean_t(max(log_softmax(logit_t)))

       This approximates −mean log P(token_t | prefix_t) over the response tokens
       under the greedy-decoding assumption.  Higher lookahead energy means the
       model assigned lower probability to the tokens it actually emitted —
       another signal of uncertainty or hallucination.

    Key classes:
    - SpilledEnergyResult: dataclass with per_token_spilled, mean_spilled,
      max_spilled, p95_spilled, lookahead_energy, suspected_hallucination,
      threshold_used, and to_dict() for JSON export.
    - compute_spilled_energy(): core function, accepts (T, V) float64 logit
      array, returns SpilledEnergyResult.
    - compute_lookahead_energy(): standalone function for the AR-EBM signal.
    - SpilledEnergyExtractor: thin class exposing extract_from_array() and
      extract_from_file(), suitable for use with saved .npy logit files from
      Exp 282/283.
    - VerifyRepairPipeline.verify_spilled_energy() is added in verify_repair.py.

Spec: REQ-VERIFY-076, SCENARIO-VERIFY-093, SCENARIO-VERIFY-094
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Core data type
# ---------------------------------------------------------------------------


@dataclass
class SpilledEnergyResult:
    """Result of a spilled-energy hallucination analysis.

    **Detailed explanation for engineers:**
        All numeric fields use Python floats (or numpy float64) for JSON
        compatibility.  ``per_token_spilled`` is kept as a numpy array for
        downstream analysis but is serialized to a Python list in to_dict().

        ``suspected_hallucination`` is True when ``mean_spilled > threshold_used``.
        The threshold default of 1.0 nat is a conservative starting point;
        calibrate on your corpus for production use.

    Attributes:
        per_token_spilled: 1-D float64 array of per-token spilled energy values,
            one entry per response token.  Shape: (T,).
        mean_spilled: Mean of per_token_spilled across all tokens.
        max_spilled: Maximum per-token spilled energy value.
        p95_spilled: 95th-percentile of per-token spilled energy values.
        lookahead_energy: AR-EBM lookahead energy: −mean(max(log_softmax(logit_t))).
        suspected_hallucination: True when mean_spilled > threshold_used.
        threshold_used: The threshold that was applied to produce the verdict.

    Spec: REQ-VERIFY-076
    """

    per_token_spilled: np.ndarray
    mean_spilled: float
    max_spilled: float
    p95_spilled: float
    lookahead_energy: float
    suspected_hallucination: bool
    threshold_used: float

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict.

        **Detailed explanation for engineers:**
            numpy arrays are converted to Python lists of floats so the result
            can be passed to json.dumps() without a custom encoder.

        Returns:
            Dict with all fields serialized to JSON-compatible Python types.

        Spec: REQ-VERIFY-076
        """
        return {
            "per_token_spilled": self.per_token_spilled.tolist(),
            "mean_spilled": float(self.mean_spilled),
            "max_spilled": float(self.max_spilled),
            "p95_spilled": float(self.p95_spilled),
            "lookahead_energy": float(self.lookahead_energy),
            "suspected_hallucination": bool(self.suspected_hallucination),
            "threshold_used": float(self.threshold_used),
        }

    def to_json(self) -> str:
        """Serialize to a JSON string with sorted keys for determinism."""
        return json.dumps(self.to_dict(), sort_keys=True)


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over last axis.

    **Detailed explanation for engineers:**
        Subtracts max before exponentiating to prevent overflow, then computes
        log(sum(exp(logits - max))) using logsumexp identity.  This is equivalent
        to scipy.special.log_softmax but avoids the scipy dependency.

    Args:
        logits: Array of shape (..., V) where V is vocab size.

    Returns:
        log-softmax values, same shape as input.
    """
    # Subtract row-wise max for numerical stability (log-sum-exp trick).
    max_logits = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - max_logits
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return np.asarray(shifted - log_sum_exp)


def compute_spilled_energy(
    logits: np.ndarray,
    threshold: float = 1.0,
) -> SpilledEnergyResult:
    """Compute per-token spilled energy from a logit array.

    **Detailed explanation for engineers:**
        Formula (per token t):
            log_probs_t = log_softmax(logit_t)          shape: (V,)
            probs_t     = exp(log_probs_t)               shape: (V,)
            H_t         = −sum(probs_t * log_probs_t)   (Shannon entropy, nats)
            max_lp_t    = max(log_probs_t)               (log prob of top token)
            spill_t     = H_t − (−max_lp_t)             = H_t + max_lp_t

        Intuition: for a perfectly peaked distribution (one token has all mass),
        H_t = 0 and max_lp_t = 0, giving spill_t = 0 — no energy spilled.
        For a uniform distribution over V tokens, H_t = log(V) and
        max_lp_t = −log(V), giving spill_t = 0 again.  The spill peaks for
        distributions that are somewhere in between — moderate uncertainty
        combined with high entropy, as often seen in hallucinating generations.

        Note: spill can be near-zero for BOTH very confident and very uncertain
        (uniform) distributions.  It captures the regime where the model "spills"
        probability mass across many tokens while still assigning non-trivial
        probability to the top token.

    Args:
        logits: 2-D float64 array of shape (T, V), where T is the number of
            response tokens and V is vocabulary size.
        threshold: Mean spilled energy above which ``suspected_hallucination``
            is set to True.  Default 1.0 nat.

    Returns:
        SpilledEnergyResult with all statistics populated.

    Raises:
        ValueError: If logits is not 2-D or has fewer than 1 token.

    Spec: REQ-VERIFY-076, SCENARIO-VERIFY-093, SCENARIO-VERIFY-094
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be 2-D (T, V), got shape {logits.shape}"
        )
    if logits.shape[0] < 1:
        raise ValueError("logits must contain at least one token (T >= 1)")

    log_probs = _log_softmax(logits)        # (T, V)
    probs = np.exp(log_probs)               # (T, V)

    # Shannon entropy per token: H_t = −sum_v(p_v * log_p_v)
    # Numerically: replace 0*log(0) → 0 (handled automatically since log_probs
    # is −inf where probs≈0, but probs≈0 * (−inf) → 0 via IEEE754 when probs=0).
    entropy = -np.sum(probs * log_probs, axis=-1)   # (T,)

    # Max log-prob per token (log prob of the greedy token).
    max_log_prob = np.max(log_probs, axis=-1)        # (T,)  ≤ 0

    # Spilled energy: entropy + max_log_prob.
    # For peaked distributions both are ~0.  For flat distributions they cancel.
    # The "spill" represents probability mass that the model spread across tokens
    # beyond what the top token absorbed.
    per_token_spilled = entropy + max_log_prob       # (T,)

    mean_spilled = float(np.mean(per_token_spilled))
    max_spilled = float(np.max(per_token_spilled))
    p95_spilled = float(np.percentile(per_token_spilled, 95))

    # AR-EBM lookahead energy: −mean(max log_softmax) ≥ 0
    lookahead_energy = float(-np.mean(max_log_prob))

    return SpilledEnergyResult(
        per_token_spilled=per_token_spilled,
        mean_spilled=mean_spilled,
        max_spilled=max_spilled,
        p95_spilled=p95_spilled,
        lookahead_energy=lookahead_energy,
        suspected_hallucination=mean_spilled > threshold,
        threshold_used=float(threshold),
    )


def compute_lookahead_energy(logits: np.ndarray) -> float:
    """Compute the AR-EBM lookahead energy from a logit array.

    **Detailed explanation for engineers:**
        From arXiv 2512.15605 (AR-EBM bijection): LLMs implicitly compute a
        "lookahead energy" in function space that equals the negative log-
        probability of the generated sequence under the model.

        Approximation used here (greedy / argmax decoding assumed):
            lookahead = −mean_t( max_v( log_softmax(logit_t) ) )

        This is the average negative log-probability assigned to the top token
        at each position.  Higher values mean the model was less certain about
        the tokens it generated — a hallucination risk signal.

        In exact AR-EBM theory the energy uses the actually-sampled token index,
        but without token IDs we use the greedy approximation (max over vocab).

    Args:
        logits: 2-D float64 array of shape (T, V).

    Returns:
        Scalar lookahead energy (≥ 0.0, in nats).

    Raises:
        ValueError: If logits is not 2-D or has fewer than 1 token.

    Spec: REQ-VERIFY-076
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be 2-D (T, V), got shape {logits.shape}"
        )
    if logits.shape[0] < 1:
        raise ValueError("logits must contain at least one token (T >= 1)")

    log_probs = _log_softmax(logits)         # (T, V)
    max_log_prob = np.max(log_probs, axis=-1)  # (T,)  ≤ 0
    return float(-np.mean(max_log_prob))


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------


class SpilledEnergyExtractor:
    """Extract a SpilledEnergyResult from a logit array or saved .npy file.

    **Detailed explanation for engineers:**
        This class is a thin wrapper around ``compute_spilled_energy()`` that
        provides two convenient entry points:

        - ``extract_from_array(logits, threshold)``: accepts a (T, V) numpy
          array directly, useful when logits are already in memory.
        - ``extract_from_file(path, threshold)``: loads a .npy file (as saved
          by Exp 282/283 logit-saving hooks) and delegates to extract_from_array.

        The class does not maintain state between calls — every call is
        independent, making it safe for concurrent use.

    Spec: REQ-VERIFY-076
    """

    DEFAULT_THRESHOLD: float = 1.0
    """Default hallucination threshold in nats (conservative starting point)."""

    def extract_from_array(
        self,
        logits: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> SpilledEnergyResult:
        """Run spilled energy analysis on a logit array already in memory.

        Args:
            logits: 2-D float64 array of shape (T, V).
            threshold: Hallucination threshold (mean spilled energy in nats).

        Returns:
            SpilledEnergyResult with all statistics.

        Spec: REQ-VERIFY-076
        """
        return compute_spilled_energy(logits, threshold=threshold)

    def extract_from_file(
        self,
        path: str | Path,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> SpilledEnergyResult:
        """Load a .npy logit file and run spilled energy analysis.

        **Detailed explanation for engineers:**
            Logit files are saved by the Exp 282/283 logit-saving hooks as
            numpy .npy files with shape (T, V).  This method loads the file
            using np.load() and delegates to extract_from_array().

        Args:
            path: File path to a .npy file containing a (T, V) logit array.
            threshold: Hallucination threshold (mean spilled energy in nats).

        Returns:
            SpilledEnergyResult with all statistics.

        Raises:
            FileNotFoundError: If the .npy file does not exist.
            ValueError: If the loaded array is not 2-D.

        Spec: REQ-VERIFY-076
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Logit file not found: {path}")
        logits = np.load(path)
        return self.extract_from_array(logits, threshold=threshold)
