"""JEPA fast-path gate — skip full Ising verification for low-risk responses.

**Researcher summary:**
    Wraps the ONNX-exported JEPA MLP (trained in Exp 307) as a lightweight
    energy gate.  If the predicted energy for a response's partial logit
    distribution falls below a threshold, we skip the expensive Ising
    verification pass entirely.  This is the primary latency-reduction
    mechanism for Exp 308: target skip_rate ≥ 30% at TP_rate ≥ 0.85.

**Detailed explanation for engineers:**
    The JEPA predictor was trained on (partial_logit_mean, violation_label)
    pairs from real Apple adversarial inference runs (Exps 294/295).  Its
    output is a scalar energy: low energy → model is confident and
    internally consistent → safe to skip Ising.  High energy → possible
    hallucination or constraint violation → run full verification.

    The gate is intentionally conservative: we only skip when energy is
    BELOW the threshold (low risk).  This preserves recall on real
    violations at the cost of some false negatives (skipping borderline
    cases).  The threshold sweep in Exp 308 finds the operating point that
    satisfies the spec target: skip_rate ≥ 0.30 AND TP_rate ≥ 0.85.

    Design choices:
    - **Lazy ONNX load**: The InferenceSession is created on the first
      call to ``predict()``, not at construction time.  This avoids
      startup overhead (onnxruntime import + model parsing ≈ 200 ms) when
      the gate is disabled or the pipeline is used without gating.
    - **disabled mode**: When ``enabled=False``, ``predict()`` returns 1.0
      (maximum energy) unconditionally, which means ``should_skip()``
      always returns False.  This makes the gate a transparent no-op when
      disabled, without changing any downstream logic.
    - **sigmoid(raw_output)**: The ONNX model emits a raw scalar; we apply
      sigmoid so the gate energy is always in [0, 1].  Values near 0 mean
      "very low risk" (model is confident); values near 1 mean "high risk".
    - **to_dict()**: Serialisation for experiment artifacts — all threshold
      sweep results embed the gate config alongside metrics.

Spec: REQ-JEPA-005, SCENARIO-JEPA-010, SCENARIO-JEPA-011
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Feature-vector fast-path predictor (exp2525 integration, REQ-JEPA-002)
# ---------------------------------------------------------------------------


_HEDGE_TERMS = frozenset(
    {
        "about",
        "apparently",
        "approximately",
        "around",
        "could",
        "guess",
        "likely",
        "maybe",
        "might",
        "perhaps",
        "possibly",
        "probable",
        "probably",
        "seem",
        "seems",
        "uncertain",
        "unclear",
        "unsure",
    }
)

_HEDGE_PHRASES = (
    "i think",
    "i guess",
    "not sure",
    "could be",
    "might be",
    "may be",
    "it depends",
)

_CONTRADICTION_PHRASES = (
    "although",
    "but",
    "cannot both",
    "correct and incorrect",
    "even though",
    "however",
    "nevertheless",
    "on the other hand",
    "true and false",
    "yet",
)

_ARITHMETIC_EQUATION_RE = re.compile(
    r"(?P<a>-?\d+)\s*(?P<op>[+\-*/xX])\s*(?P<b>-?\d+)\s*=\s*(?P<result>-?\d+)"
)


def _count_phrases(text_lower: str, phrases: tuple[str, ...]) -> int:
    """Count lightweight cue phrases without relying on an NLP dependency."""
    return sum(1 for phrase in phrases if phrase in text_lower)


def _arithmetic_inconsistency(response: str) -> float:
    """Return 1.0 when a simple integer equation in the response is wrong."""
    for match in _ARITHMETIC_EQUATION_RE.finditer(response):
        a = int(match.group("a"))
        b = int(match.group("b"))
        stated = int(match.group("result"))
        op = match.group("op")
        if op == "+":
            expected = a + b
        elif op == "-":
            expected = a - b
        elif op in {"*", "x", "X"}:
            expected = a * b
        elif b != 0 and a % b == 0:
            expected = a // b
        else:
            continue
        if expected != stated:
            return 1.0
    return 0.0


def extract_response_features(response: str) -> dict[str, float]:
    """Extract lightweight violation-predictive features from response text.

    **Detailed explanation for engineers:**
        Produces proxy signals without requiring LLM logprobs — important
        because logprobs are only available when an LLM is loaded in the
        pipeline. The first two signals correlate with the exp2525 feature set
        (ising_energy_response_level and logprob_variance), while the text-cue
        signals keep obviously risky short responses out of the fast path:

        1. ``response_length_norm``: normalised token count.
           Computed as ``len(response.split()) / 200``, capped at 1.0.
           Short numeric answers (e.g. "75") score near 0.0; multi-sentence
           reasoning passages score near 1.0.  Arithmetic violations typically
           appear in short answers that get caught even at low length.

        2. ``logprob_variance_proxy``: 1 − (bigram_entropy / max_entropy).
           High bigram entropy = many diverse bigrams = natural text (lower
           fabrication risk). Low entropy = repetitive bigrams = monotone /
           possibly collapsed output.  Ranges 0.0 (maximum diversity) to 1.0
           (single repeated bigram).
        3. ``hedge_cue_norm``: bounded count of uncertainty terms such as
           "maybe", "might", or "not sure".
        4. ``contradiction_cue_norm``: bounded count of contrast or direct
           contradiction phrases such as "but" or "correct and incorrect".
        5. ``arithmetic_inconsistency``: exact check for simple integer
           equations like ``5 + 7 = 13``.

    Args:
        response: The response text to featurise.

    Returns:
        Dict with bounded feature values in [0.0, 1.0].
    """
    tokens = response.split()
    response_lower = response.lower()
    word_tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", response_lower)

    # Feature 1: length normalised to [0, 1] using 200-token reference scale
    length_norm = min(len(tokens) / 200.0, 1.0)

    # Feature 2: bigram-entropy proxy (inverse of diversity = variance analog)
    if len(response) > 3:
        bigrams = [response[i : i + 2] for i in range(len(response) - 1)]
        counts = Counter(bigrams)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p + 1e-12) for p in probs)
        max_ent = math.log2(max(len(counts), 1))
        # Variance proxy = distance from maximum-entropy (uniform bigram distribution)
        variance_proxy = 1.0 - (entropy / max_ent if max_ent > 0.0 else 0.0)
        variance_proxy = max(0.0, min(1.0, variance_proxy))
    else:
        variance_proxy = 0.0

    hedge_count = sum(1 for token in word_tokens if token in _HEDGE_TERMS)
    hedge_count += _count_phrases(response_lower, _HEDGE_PHRASES)
    hedge_cue_norm = min(hedge_count / 3.0, 1.0)

    contradiction_count = _count_phrases(response_lower, _CONTRADICTION_PHRASES)
    contradiction_cue_norm = min(contradiction_count / 2.0, 1.0)

    return {
        "response_length_norm": length_norm,
        "logprob_variance_proxy": variance_proxy,
        "hedge_cue_norm": hedge_cue_norm,
        "contradiction_cue_norm": contradiction_cue_norm,
        "arithmetic_inconsistency": _arithmetic_inconsistency(response),
    }


@dataclass
class JEPAFastPathPredictor:
    """Feature-based P(violation) predictor gating Ising verification.

    **Researcher summary:**
        Implements the Tier 3 JEPA fast-path from exp2525 (AUC=0.8889).
        Predicts P(constraint violation) from two lightweight text-derived
        proxy features so that clearly non-violating responses can bypass
        expensive Ising verification.  Target threshold: p_violation < 0.2.

    **Detailed explanation for engineers:**
        Wired into ``VerifyRepairPipeline`` via the ``jepa_fast_path_predictor``
        constructor argument.  ``predict_p_violation`` is called at the top
        of ``verify()`` before extraction or Ising computation.  When the
        prediction is below the pipeline's ``jepa_fast_path_threshold`` (default
        0.2), ``verify()`` returns a fast-path ``VerificationResult`` with
        ``verified=True`` and the certificate key ``skipped_verification=True``.

        Each call increments ``calls_total``; the pipeline increments
        ``calls_fast_path`` when a fast-path is taken.  Use ``fast_path_rate``
        to measure the hit rate across a test batch.

        Decision rule (derived from exp2525 feature importance):
            - Both features < 0.1 → p = 0.05 (very short, uniform text such
              as a single numeric answer — almost certainly non-violating).
            - Otherwise → p = 0.4 * length_norm + 0.6 * variance_proxy.
              Variance contributes more than length because a short but
              low-entropy response can still be wrong.

    Example::

        predictor = JEPAFastPathPredictor()
        p = predictor.predict_p_violation("The answer is 75.")
        if p < 0.2:
            return fast_path_result(verified=True)

    Spec: REQ-VERIFY-003, REQ-JEPA-002
    """

    calls_total: int = field(default=0, init=False)
    calls_fast_path: int = field(default=0, init=False)

    def predict_p_violation(self, response: str) -> float:
        """Predict P(constraint violation) from response text features.

        **Detailed explanation for engineers:**
            Calls ``extract_response_features`` to get length_norm and
            variance_proxy, then applies the two-stage decision rule:

            1. If hedging, contradiction, or arithmetic inconsistency cues are
               present, return at least 0.25 so the response cannot slip below
               the default 0.2 fast-path threshold merely because it is short.
            2. If BOTH legacy features < 0.1 (very short, compositionally
               uniform text like "42" or "yes"): return 0.05 — fast-path
               eligible.
            3. Otherwise: return 0.4 * length_norm + 0.6 * variance_proxy.

            The threshold check against the pipeline's
            ``jepa_fast_path_threshold`` (default 0.2) happens in
            ``VerifyRepairPipeline.verify()``, not here — this method
            only computes the probability.

        Args:
            response: The response text to score.

        Returns:
            Float P(violation) in [0.0, 1.0].  Values below 0.2 are
            considered low-risk by the default pipeline threshold.

        Spec: REQ-JEPA-002
        """
        self.calls_total += 1
        features = extract_response_features(response)
        length_norm = features["response_length_norm"]
        variance_proxy = features["logprob_variance_proxy"]
        hedge_cue_norm = features["hedge_cue_norm"]
        contradiction_cue_norm = features["contradiction_cue_norm"]
        arithmetic_inconsistency = features["arithmetic_inconsistency"]

        if max(hedge_cue_norm, contradiction_cue_norm, arithmetic_inconsistency) > 0.0:
            p = (
                0.25
                + 0.35 * hedge_cue_norm
                + 0.30 * contradiction_cue_norm
                + 0.35 * arithmetic_inconsistency
                + 0.05 * length_norm
                + 0.05 * variance_proxy
            )
            return min(max(p, 0.0), 1.0)

        # Both signals negligible → response is very short and uniform
        # (e.g. a single numeric answer). Near-zero violation risk.
        if length_norm < 0.1 and variance_proxy < 0.1:
            return 0.05

        p = 0.4 * length_norm + 0.6 * variance_proxy
        return min(max(p, 0.0), 1.0)

    def predict(self, response: str) -> float:
        """Compatibility alias for Exp 2550 text-corpus scoring.

        The ONNX-backed ``JepaGate`` already exposes ``predict(logit_mean)``.
        This feature-vector predictor now mirrors that naming for callers that
        only have response text, while preserving ``predict_p_violation`` as
        the explicit probability API used by ``VerifyRepairPipeline``.

        Spec: REQ-JEPA-006
        """
        return self.predict_p_violation(response)

    @property
    def fast_path_rate(self) -> float:
        """Fraction of calls where fast-path was used (as logged by the pipeline).

        Returns float("nan") when no calls have been made to avoid ZeroDivisionError.
        """
        if self.calls_total == 0:
            return float("nan")
        return self.calls_fast_path / self.calls_total


# ---------------------------------------------------------------------------
# JepaGate
# ---------------------------------------------------------------------------


@dataclass
class JepaGate:
    """Fast-path energy gate backed by the ONNX JEPA predictor.

    **Detailed explanation for engineers:**
        Wraps a trained JEPA MLP (exported as ONNX by Exp 291/307) as a
        lightweight gate.  The gate receives the mean logit vector for a
        response, runs one ONNX forward pass, and returns a scalar energy
        in [0, 1] via sigmoid.  If that energy is below ``threshold`` the
        response is deemed low-risk and the full Ising verification step is
        skipped (``should_skip`` returns True).

        The gate is safe to instantiate even when the ONNX file is absent
        or onnxruntime is not installed — construction never raises.  Errors
        surface only when ``predict()`` is called, so callers can build the
        gate once at startup and handle failures at call time.

    Attributes:
        onnx_path: Path to the trained ONNX model file.
        threshold: Energy below which the gate fires (skip Ising).
            Default 0.5.  Lower → more aggressive skipping, higher miss risk.
        enabled: When False, ``predict()`` always returns 1.0 (no skipping).
            Default True.

    Spec: REQ-JEPA-005
    """

    onnx_path: str
    threshold: float = 0.5
    enabled: bool = True

    # Private: lazily-created ONNX session — not part of the public interface.
    _session: Any = field(default=None, init=False, repr=False, compare=False)

    def _get_session(self) -> Any:
        """Load and cache the ONNX InferenceSession on first use.

        **Detailed explanation for engineers:**
            onnxruntime import + model parsing costs ≈ 200 ms the first time.
            We defer it to the first ``predict()`` call so pipelines that
            don't use the gate pay zero startup cost.

        Returns:
            onnxruntime.InferenceSession, cached after first call.

        Raises:
            ImportError: If onnxruntime is not installed.
            OSError: If the ONNX file does not exist.
        """
        if self._session is None:
            try:
                import onnxruntime as ort  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required for JepaGate.predict(). "
                    "Install with: pip install onnxruntime"
                ) from exc
            # Silence onnxruntime logging noise (INFO-level provider selection).
            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # ERROR only
            self._session = ort.InferenceSession(str(self.onnx_path), sess_options=opts)
        return self._session

    def predict(self, logit_mean: np.ndarray) -> float:
        """Predict hallucination risk energy from a partial logit mean vector.

        **Detailed explanation for engineers:**
            Runs the ONNX forward pass on a 1-D numpy array of mean logit
            values (shape (V,) where V = vocab_size or a reduced feature
            dimension, depending on which ONNX model is loaded).

            The raw scalar output from the model is passed through sigmoid
            to normalise it into [0, 1].  The interpretation is:
            - Near 0 → model is internally consistent, low hallucination risk.
            - Near 1 → high predicted energy, Ising verification recommended.

            When ``enabled=False``, returns 1.0 immediately — equivalent to
            maximum energy, which ensures ``should_skip()`` returns False
            and the full pipeline always runs.

        Args:
            logit_mean: 1-D numpy array, mean logit values for the response.
                Must be compatible with the ONNX model's expected input shape.

        Returns:
            Float in [0, 1]: sigmoid(raw ONNX output).  Lower = safer to skip.

        Raises:
            ImportError: If onnxruntime is not installed.
            OSError: If the ONNX file does not exist.

        Spec: REQ-JEPA-005
        """
        if not self.enabled:
            # Gate disabled — always return maximum energy so we never skip.
            return 1.0

        import numpy as np  # local import for lazy dependency

        session = self._get_session()
        input_name = session.get_inputs()[0].name
        arr = np.asarray(logit_mean, dtype=np.float32).reshape(1, -1)
        raw_output = session.run(None, {input_name: arr})[0]
        raw_scalar = float(np.asarray(raw_output).flat[0])
        # Apply sigmoid to map raw energy to [0, 1].
        energy = 1.0 / (1.0 + math.exp(-raw_scalar))
        return energy

    def should_skip(self, logit_mean: np.ndarray) -> bool:
        """Decide whether to skip full Ising verification for this response.

        **Detailed explanation for engineers:**
            Returns True when ``predict(logit_mean) < threshold``, meaning
            the model's internal energy is below the risk threshold and
            full Ising verification can be safely bypassed.

            Callers should use this method rather than calling ``predict()``
            and comparing manually, because it handles the disabled-gate case
            (returns False when enabled=False, preserving the full pipeline).

        Args:
            logit_mean: 1-D numpy array, mean logit values for the response.

        Returns:
            True if Ising can be skipped; False if full verification is needed.

        Spec: REQ-JEPA-005, SCENARIO-JEPA-010, SCENARIO-JEPA-011
        """
        if not self.enabled:
            return False
        energy = self.predict(logit_mean)
        return energy < self.threshold

    def to_dict(self) -> dict[str, object]:
        """Serialise gate config for experiment artifacts.

        Returns:
            Dict with keys ``onnx_path`` (str), ``threshold`` (float),
            ``enabled`` (bool).  Safe to embed in JSON result artifacts.
        """
        return {
            "onnx_path": str(self.onnx_path),
            "threshold": self.threshold,
            "enabled": self.enabled,
        }
