"""Tier 3 predictive gate — estimate whether a partial response will trigger
downstream verification violations before running the full (expensive) pipeline.

**Researcher summary:**
    A lightweight NumPy-backed logistic gate sits in front of the full EBM
    verification path. It converts a partial response into a small feature
    vector (token counts, numeric density, operator presence, JSON structure),
    applies a calibrated linear model, and outputs a routing decision:

    - ``FAST_PATH``: predicted violation probability is below the threshold —
      skip full verification (optimistic low-risk assumption).
    - ``FULL``: predicted probability meets or exceeds the threshold — proceed
      with the expensive constraint extraction + Ising pass.

    The gate is intentionally simpler than the JEPA MLP in ``jepa_predictor.py``
    so it can be exported to ONNX or run on an NPU without a JAX install.

**Detailed explanation for engineers:**
    Architecture:

    1. **Feature extraction** (``extract_features``): Pure Python + NumPy.
       Parses the partial response text into nine scalar features packed into
       a ``PredictiveFeatures`` dataclass.  Features are:

       - ``token_count``      — number of whitespace-split tokens (0..∞)
       - ``char_count``       — raw character length (0..∞)
       - ``numeric_density``  — fraction of tokens that look numeric (0..1)
       - ``operator_density`` — fraction of tokens that are arith operators
                                (+, -, *, /, =) (0..1)
       - ``json_parseable``   — 1.0 if text parses as JSON, else 0.0
       - ``n_claims``         — number of items in a ``claims`` JSON array
                                (0 if not JSON or key absent)
       - ``has_final_answer`` — 1.0 if a ``final_answer`` key is present
                                in parsed JSON, else 0.0
       - ``domain_code``      — 0.0 for "reasoning" domain, 1.0 otherwise
       - ``prior_confidence`` — caller-supplied confidence hint (0..1)

    2. **Gate model** (``PredictiveVerifier``): Logistic regression stored as
       a weight vector ``w`` (shape FEATURE_DIM) and a scalar bias ``b``.
       ``confidence = sigmoid(w @ x + b)`` where ``x`` is the normalised
       feature vector.  Default weights produce a conservative mid-range prior
       (threshold 0.5 → borderline cases go FULL).

    3. **Calibration** (``calibrate``): Accepts corpus rows from the Exp 252
       predictive-verification corpus and updates ``w`` / ``b`` via one pass
       of logistic gradient descent — no JAX or heavy ML libraries required.

    4. **ONNX export** (``export_onnx``): Converts the linear gate to a
       minimal ONNX graph (MatMul + Add + Sigmoid) so the model can run on
       any ONNX runtime, including NPU / CUDA accelerators, without Python.

    5. **Duck-type compatibility** (``predict_embedding``): Satisfies the
       ``jepa_predictor`` duck-type interface expected by
       ``VerifyRepairPipeline.verify()``, so a ``PredictiveVerifier`` instance
       can be passed directly as the ``jepa_predictor`` argument.

    6. **Serialisation**: Gate weights persist to / from a ``.safetensors``
       file (two tensors: ``w`` and ``b``) for deterministic checkpoint replay.

Spec: REQ-PRED-001, REQ-PRED-002, REQ-PRED-003, REQ-PRED-004
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

RUN_DATE: str = "20260413"
"""Fixed run-date embedded in every result record for traceability."""

ROUTE_FAST_PATH: str = "FAST_PATH"
"""Route literal: skip full verification (low-risk optimistic assumption)."""

ROUTE_FULL: str = "FULL"
"""Route literal: proceed with full constraint extraction + Ising pass."""

# The feature vector has exactly FEATURE_DIM elements (in order):
#   token_count, char_count, numeric_density, operator_density,
#   json_parseable, n_claims, has_final_answer, domain_code, prior_confidence
FEATURE_DIM: int = 9
"""Fixed input dimensionality of the gate's feature vector."""

# Domain names emitted in domain_probs (mirrors jepa_predictor.DOMAINS).
_GATE_DOMAINS: list[str] = ["arithmetic", "code", "logic"]

# Pattern to detect arithmetic operator tokens.
_OPERATOR_RE = re.compile(r"^[+\-*/=]$")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictiveFeatures:
    """Structured scalar features extracted from a partial response.

    **Detailed explanation for engineers:**
        All nine features are plain Python scalars so the dataclass is safe
        to construct in any environment.  ``to_array()`` packs them into a
        NumPy float32 vector of length ``FEATURE_DIM`` for the gate model.

    Attributes:
        token_count:      Number of whitespace-split tokens.
        char_count:       Raw character length of the partial response.
        numeric_density:  Fraction of tokens that look numeric (0..1).
        operator_density: Fraction of tokens that are arith operators (0..1).
        json_parseable:   1.0 if text parses as JSON, else 0.0.
        n_claims:         Number of items in ``claims`` JSON array (0 if absent).
        has_final_answer: 1.0 if ``final_answer`` key present in JSON, else 0.0.
        domain_code:      0.0 for "reasoning" domain, 1.0 for any other.
        prior_confidence: Caller-supplied confidence hint in [0, 1].
        run_date:         Fixed string ``"20260413"`` for traceability.

    Spec: REQ-PRED-001
    """

    token_count: int
    char_count: int
    numeric_density: float
    operator_density: float
    json_parseable: float
    n_claims: int
    has_final_answer: float
    domain_code: float
    prior_confidence: float
    run_date: str = RUN_DATE

    def to_array(self) -> np.ndarray:
        """Return a float32 NumPy array of shape ``(FEATURE_DIM,)``.

        **Detailed explanation for engineers:**
            Values are packed in the fixed order listed in the module docstring.
            ``token_count`` and ``n_claims`` are divided by a scale factor to
            keep them in a comparable range to the [0,1]-bounded features,
            which stabilises the logistic gate without requiring per-feature
            standardisation.

        Returns:
            1-D float32 array of length ``FEATURE_DIM`` = 9.

        Spec: REQ-PRED-001
        """
        return np.array(
            [
                min(self.token_count / 100.0, 1.0),  # scaled to ~[0,1]
                min(self.char_count / 500.0, 1.0),   # scaled to ~[0,1]
                self.numeric_density,
                self.operator_density,
                self.json_parseable,
                min(self.n_claims / 10.0, 1.0),      # scaled to ~[0,1]
                self.has_final_answer,
                self.domain_code,
                self.prior_confidence,
            ],
            dtype=np.float32,
        )


def extract_features(
    partial_response: str,
    domain: str | None = None,
    prior_confidence: float = 0.5,
) -> PredictiveFeatures:
    """Extract structured features from a partial response string.

    **Detailed explanation for engineers:**
        This is a pure Python + NumPy function — no JAX, PyTorch, or network
        calls.  It is safe to import and call on any CPU-only deployment.

        JSON detection: ``json.loads`` is tried first.  If it succeeds, the
        ``claims`` and ``final_answer`` features are populated from the parsed
        dict.  A partially-valid JSON string (e.g. truncated streaming output)
        is treated as non-parseable.

        Operator detection: tokens consisting entirely of ``+ - * / =``
        characters count as operators.  This captures arithmetic steps like
        ``"55 * 4 = 220"`` without false-positive on ``e-mail`` etc.

        Numeric detection: a token is numeric if ``str.isdigit()`` returns
        True after stripping leading/trailing punctuation, or if the token
        matches an optional sign followed by digits and at most one decimal
        point.

    Args:
        partial_response: The first N tokens of a model response (any string).
        domain: Optional domain hint.  ``"reasoning"`` → ``domain_code=0.0``;
            any other non-None value → ``domain_code=1.0``.  ``None`` (default)
            defaults to ``domain_code=0.0`` (reasoning assumed).
        prior_confidence: Caller-supplied confidence estimate in [0, 1].
            Defaults to 0.5 (no prior opinion).

    Returns:
        ``PredictiveFeatures`` with all nine fields populated.

    Spec: REQ-PRED-001
    """
    text = partial_response or ""
    tokens = text.split()
    token_count = len(tokens)
    char_count = len(text)

    # Numeric and operator density over whitespace tokens.
    _numeric_re = re.compile(r"^[+-]?\d+(\.\d+)?$")
    if token_count > 0:
        stripped = [t.strip(".,;:!?()'\"") for t in tokens]
        numeric_count = sum(1 for t in stripped if _numeric_re.match(t))
        operator_count = sum(1 for t in stripped if _OPERATOR_RE.match(t))
        numeric_density = float(numeric_count) / token_count
        operator_density = float(operator_count) / token_count
    else:
        numeric_density = 0.0
        operator_density = 0.0

    # JSON structure detection.
    json_parseable = 0.0
    n_claims = 0
    has_final_answer = 0.0
    if text.strip():
        try:
            parsed = json.loads(text)
            json_parseable = 1.0
            if isinstance(parsed, dict):
                claims = parsed.get("claims")
                if isinstance(claims, list):
                    n_claims = len(claims)
                if "final_answer" in parsed:
                    has_final_answer = 1.0
        except (json.JSONDecodeError, ValueError):
            pass

    # Domain encoding: 0.0 = "reasoning" (default), 1.0 = anything else.
    if domain is None or domain == "reasoning":
        domain_code = 0.0
    else:
        domain_code = 1.0

    return PredictiveFeatures(
        token_count=token_count,
        char_count=char_count,
        numeric_density=numeric_density,
        operator_density=operator_density,
        json_parseable=json_parseable,
        n_claims=n_claims,
        has_final_answer=has_final_answer,
        domain_code=domain_code,
        prior_confidence=float(np.clip(prior_confidence, 0.0, 1.0)),
        run_date=RUN_DATE,
    )


# ---------------------------------------------------------------------------
# Gate decision
# ---------------------------------------------------------------------------


@dataclass
class GateDecision:
    """Calibrated routing decision from the predictive gate.

    **Detailed explanation for engineers:**
        ``should_skip=True`` means the gate predicts the partial response is
        *low-risk* and the full verification pass can be skipped.
        ``should_skip=False`` means the gate predicts a meaningful downstream
        violation is likely and the expensive path should run.

        The ``confidence`` field is the gate's raw sigmoid output — a float
        in [0, 1] representing *predicted violation probability*.  High
        confidence = high risk = ``FULL`` path; low confidence = low risk =
        ``FAST_PATH``.

        The ``domain_probs`` dict replicates the interface expected by
        ``VerifyRepairPipeline.verify(jepa_predictor=...)`` so callers can
        treat a ``PredictiveVerifier`` like a ``JEPAViolationPredictor``
        without code changes.

    Attributes:
        should_skip:     True when the gate predicts low violation risk.
        confidence:      Predicted violation probability in [0, 1].
        threshold:       The threshold that was applied to make the decision.
        route:           ``"FAST_PATH"`` or ``"FULL"``.
        domain_probs:    Per-domain violation probability estimates (dict).
        feature_summary: Human-readable feature values for audit/debugging.
        run_date:        Fixed string ``"20260413"`` for traceability.

    Spec: REQ-PRED-002
    """

    should_skip: bool
    confidence: float
    threshold: float
    route: str
    domain_probs: dict[str, float] = field(default_factory=dict)
    feature_summary: dict[str, Any] = field(default_factory=dict)
    run_date: str = RUN_DATE

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministically sorted dict.

        **Detailed explanation for engineers:**
            All keys are sorted so that two calls on the same ``GateDecision``
            produce identical output — required for audit-trail reproducibility
            and for embedding the decision in ``VerificationResult.certificate``.

        Returns:
            Dict with sorted string keys and JSON-serialisable values.

        Spec: REQ-PRED-002
        """
        raw: dict[str, Any] = {
            "confidence": self.confidence,
            "domain_probs": {k: self.domain_probs[k] for k in sorted(self.domain_probs)},
            "feature_summary": {
                k: self.feature_summary[k] for k in sorted(self.feature_summary)
            },
            "route": self.route,
            "run_date": self.run_date,
            "should_skip": self.should_skip,
            "threshold": self.threshold,
        }
        return dict(sorted(raw.items()))

    def to_json(self) -> str:
        """Deterministic JSON serialisation.

        Returns:
            JSON string with sorted keys, no trailing whitespace.

        Spec: REQ-PRED-002
        """
        return json.dumps(self.to_dict(), sort_keys=True)


# ---------------------------------------------------------------------------
# Predictive verifier
# ---------------------------------------------------------------------------

# Default weight vector and bias for the logistic gate.
# These encode a mild prior: numeric-heavy + JSON responses carry ~60% risk.
# See _DEFAULT_W comments for each feature's intended contribution.
_DEFAULT_W = np.array(
    [
        0.3,   # token_count (scaled): more tokens → slightly more content to check
        0.1,   # char_count (scaled): weakly correlated with numeric density
        1.2,   # numeric_density: strong positive signal for arithmetic violations
        0.8,   # operator_density: arithmetic operators → high risk
        0.5,   # json_parseable: structured output is usually monitorable → mild risk
        0.4,   # n_claims (scaled): more claims → more to verify
        0.3,   # has_final_answer: explicit answer declaration → more checkable
        0.2,   # domain_code: code domain slightly higher risk than reasoning
        0.6,   # prior_confidence: caller hint dominates when strong
    ],
    dtype=np.float32,
)
_DEFAULT_B = np.float32(-1.5)  # negative bias → low baseline risk

# Learning rate for one-pass calibration gradient descent.
_CALIBRATE_LR = 0.05


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class PredictiveVerifier:
    """Calibrated logistic gate for Tier 3 predictive verification.

    **Researcher summary:**
        Lightweight CPU gate that estimates downstream violation risk from
        partial-response features without loading any model.  Replaces the
        random-projection → JEPA-MLP path with a directly interpretable
        linear model whose weights can be calibrated from a live corpus.

    **Detailed explanation for engineers:**
        The gate computes ``confidence = sigmoid(w @ x + b)`` where:

        - ``x``  is the FEATURE_DIM-dimensional feature vector from
          ``PredictiveFeatures.to_array()``
        - ``w``  is the weight vector (shape ``(FEATURE_DIM,)``)
        - ``b``  is the scalar bias

        Default weights encode a conservative prior (see ``_DEFAULT_W``).
        ``calibrate(corpus_rows)`` runs one pass of logistic gradient descent
        using the ``verifier_outcome`` labels from an Exp 252 corpus slice.

        **Duck-type compatibility with JEPAViolationPredictor:**
            ``predict_embedding(embedding)`` accepts a 256-D random-projection
            embedding (the same format ``VerifyRepairPipeline.verify()`` sends
            to ``jepa_predictor``) and returns a per-domain probability dict.
            Under the hood it computes a single scalar from the embedding's
            L2 norm (a proxy for text richness) and maps it to per-domain
            probs.  This is intentionally a proxy — calibrated structured
            features from ``gate()`` are more accurate.

        **ONNX export:**
            The gate's MatMul + Add + Sigmoid computation graph is trivially
            exportable to ONNX format.  ``export_onnx(path)`` does this
            without executing the model, so the file captures the current
            weights at call time.

        **Serialisation:**
            ``save(path)`` / ``load(path)`` persist the weight vector and
            bias to/from a ``.safetensors`` file (two tensors: ``w`` and
            ``b``).

    Example::

        vp = PredictiveVerifier()
        decision = vp.gate(partial_response, threshold=0.5)
        if decision.should_skip:
            return fast_path_result()
        return full_pipeline(response)

    Spec: REQ-PRED-001, REQ-PRED-002, REQ-PRED-003, REQ-PRED-004
    """

    def __init__(self) -> None:
        """Initialise with default conservative prior weights."""
        self._w: np.ndarray = _DEFAULT_W.copy()
        self._b: np.ndarray = _DEFAULT_B.copy()

    # ------------------------------------------------------------------
    # Core gate interface
    # ------------------------------------------------------------------

    def extract_features_for(
        self,
        partial_response: str,
        domain: str | None = None,
        prior_confidence: float = 0.5,
    ) -> PredictiveFeatures:
        """Delegate to the module-level ``extract_features`` function.

        Args:
            partial_response: Partial response text.
            domain: Optional domain hint.
            prior_confidence: Caller-supplied confidence hint in [0, 1].

        Returns:
            Populated ``PredictiveFeatures`` instance.

        Spec: REQ-PRED-001
        """
        return extract_features(partial_response, domain=domain, prior_confidence=prior_confidence)

    def predict(
        self,
        features: PredictiveFeatures,
        threshold: float = 0.5,
    ) -> GateDecision:
        """Compute a calibrated gate decision from extracted features.

        **Detailed explanation for engineers:**
            Runs ``confidence = sigmoid(w @ x + b)`` on the feature array.
            If ``confidence < threshold``, the gate considers the partial
            response low-risk and returns ``FAST_PATH``.  Otherwise, returns
            ``FULL``.

            ``domain_probs`` is derived from ``confidence`` by applying a
            per-domain scaling factor (arithmetic = full confidence, logic =
            70%, code = 60%) so that the dict interface matches what
            ``VerifyRepairPipeline`` expects from a JEPA predictor.

        Args:
            features: Features from ``extract_features`` or
                ``extract_features_for``.
            threshold: Violation-probability threshold.  Default 0.5.

        Returns:
            ``GateDecision`` with routing, confidence, and metadata.

        Spec: REQ-PRED-002
        """
        x = features.to_array()
        raw = float(np.dot(self._w, x) + self._b)
        confidence = float(_sigmoid(raw))

        should_skip = confidence < threshold
        route = ROUTE_FAST_PATH if should_skip else ROUTE_FULL

        # Per-domain violation probability estimates derived from confidence.
        # Arithmetic carries the highest weight because the feature vector's
        # numeric/operator signals are most directly predictive of arithmetic
        # violations.  Code and logic are scaled down conservatively.
        domain_probs: dict[str, float] = {
            "arithmetic": float(np.clip(confidence * 1.0, 0.0, 1.0)),
            "logic":      float(np.clip(confidence * 0.7, 0.0, 1.0)),
            "code":       float(np.clip(confidence * 0.6, 0.0, 1.0)),
        }

        feature_summary: dict[str, Any] = {
            "token_count":      features.token_count,
            "char_count":       features.char_count,
            "numeric_density":  round(features.numeric_density, 4),
            "operator_density": round(features.operator_density, 4),
            "json_parseable":   features.json_parseable,
            "n_claims":         features.n_claims,
            "has_final_answer": features.has_final_answer,
            "domain_code":      features.domain_code,
            "prior_confidence": round(features.prior_confidence, 4),
        }

        return GateDecision(
            should_skip=should_skip,
            confidence=confidence,
            threshold=float(threshold),
            route=route,
            domain_probs=domain_probs,
            feature_summary=feature_summary,
            run_date=RUN_DATE,
        )

    def gate(
        self,
        partial_response: str,
        threshold: float = 0.5,
        domain: str | None = None,
        prior_confidence: float = 0.5,
    ) -> GateDecision:
        """One-shot gate: extract features then predict in a single call.

        **Detailed explanation for engineers:**
            Convenience wrapper combining ``extract_features_for`` and
            ``predict`` for callers that do not need to inspect the raw
            ``PredictiveFeatures`` object.

        Args:
            partial_response: First N tokens of the model response.
            threshold: Violation-probability threshold.  Default 0.5.
            domain: Optional domain hint.
            prior_confidence: Caller-supplied confidence hint in [0, 1].

        Returns:
            ``GateDecision`` with routing, confidence, and metadata.

        Spec: REQ-PRED-002
        """
        feats = self.extract_features_for(partial_response, domain=domain,
                                          prior_confidence=prior_confidence)
        return self.predict(feats, threshold=threshold)

    # ------------------------------------------------------------------
    # Duck-type compatibility: jepa_predictor interface
    # ------------------------------------------------------------------

    def predict_embedding(self, partial_embedding: np.ndarray) -> dict[str, float]:
        """Duck-type interface matching ``JEPAViolationPredictor.predict()``.

        **Detailed explanation for engineers:**
            ``VerifyRepairPipeline.verify()`` calls
            ``jepa_predictor.predict(embedding)`` where ``embedding`` is a
            256-D ``RandomProjectionEmbedding`` output.  This method accepts
            that same embedding and returns a per-domain probability dict so
            that a ``PredictiveVerifier`` can be used wherever a
            ``JEPAViolationPredictor`` is expected.

            The mapping is intentionally simple: we compute the L2 norm of the
            embedding (a proxy for how "information-rich" the partial response
            is) and feed it through the scalar gate as a single feature.
            This is a rough approximation — use ``gate()`` with the raw text
            when accurate routing matters.

        Args:
            partial_embedding: 1-D array of shape (256,) from
                ``RandomProjectionEmbedding.encode()``.

        Returns:
            Dict mapping domain name → float violation probability in [0, 1].

        Spec: REQ-PRED-004
        """
        arr = np.asarray(partial_embedding, dtype=np.float32).ravel()
        # Use the L2 norm (normalised by sqrt(dim)) as a single-feature proxy.
        norm = float(np.linalg.norm(arr)) / max(float(np.sqrt(arr.size)), 1.0)
        # Scale to [0, 1] via sigmoid with conservative prior.
        raw_confidence = float(_sigmoid(norm - 0.5))
        return {
            "arithmetic": float(np.clip(raw_confidence * 1.0, 0.0, 1.0)),
            "logic":      float(np.clip(raw_confidence * 0.7, 0.0, 1.0)),
            "code":       float(np.clip(raw_confidence * 0.6, 0.0, 1.0)),
        }

    # Alias so the existing jepa_predictor call-site in verify_repair.py works
    # transparently: it calls ``predictor.predict(embedding)``.  This alias
    # maps that call to ``predict_embedding`` so the duck-type contract is met
    # while keeping the primary ``predict(features)`` API on this class.
    def predict(  # type: ignore[override]
        self,
        features_or_embedding: PredictiveFeatures | np.ndarray,
        threshold: float = 0.5,
    ) -> GateDecision | dict[str, float]:
        """Overloaded predict: accepts either PredictiveFeatures or a raw embedding.

        **Detailed explanation for engineers:**
            When called with a ``PredictiveFeatures`` instance (the primary
            public API), returns a ``GateDecision``.

            When called with a NumPy array (the duck-type path used by
            ``VerifyRepairPipeline`` which passes a 256-D embedding), returns
            the ``dict[str, float]`` per-domain probability map expected by
            the pipeline.

        Args:
            features_or_embedding: Either a ``PredictiveFeatures`` object or
                a raw 1-D NumPy embedding array.
            threshold: Only used when ``features_or_embedding`` is a
                ``PredictiveFeatures`` object.

        Returns:
            ``GateDecision`` for ``PredictiveFeatures`` input, or
            ``dict[str, float]`` for raw embedding input.

        Spec: REQ-PRED-002, REQ-PRED-004
        """
        if isinstance(features_or_embedding, PredictiveFeatures):
            return self._predict_from_features(features_or_embedding, threshold)
        # Raw embedding array path (duck-type jepa_predictor compatibility).
        return self.predict_embedding(features_or_embedding)

    def _predict_from_features(
        self,
        features: PredictiveFeatures,
        threshold: float = 0.5,
    ) -> GateDecision:
        """Internal implementation of predict() for PredictiveFeatures input."""
        x = features.to_array()
        raw = float(np.dot(self._w, x) + self._b)
        confidence = float(_sigmoid(raw))

        should_skip = confidence < threshold
        route = ROUTE_FAST_PATH if should_skip else ROUTE_FULL

        domain_probs: dict[str, float] = {
            "arithmetic": float(np.clip(confidence * 1.0, 0.0, 1.0)),
            "logic":      float(np.clip(confidence * 0.7, 0.0, 1.0)),
            "code":       float(np.clip(confidence * 0.6, 0.0, 1.0)),
        }

        feature_summary: dict[str, Any] = {
            "token_count":      features.token_count,
            "char_count":       features.char_count,
            "numeric_density":  round(features.numeric_density, 4),
            "operator_density": round(features.operator_density, 4),
            "json_parseable":   features.json_parseable,
            "n_claims":         features.n_claims,
            "has_final_answer": features.has_final_answer,
            "domain_code":      features.domain_code,
            "prior_confidence": round(features.prior_confidence, 4),
        }

        return GateDecision(
            should_skip=should_skip,
            confidence=confidence,
            threshold=float(threshold),
            route=route,
            domain_probs=domain_probs,
            feature_summary=feature_summary,
            run_date=RUN_DATE,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        corpus_rows: list[dict[str, Any]],
        n_passes: int = 1,
        lr: float = _CALIBRATE_LR,
    ) -> None:
        """Update gate weights from Exp 252 corpus rows.

        **Detailed explanation for engineers:**
            Each corpus row is expected to follow the Exp 252 schema
            (``predictive_verification_corpus_252.jsonl``):

            - ``partial_response``: The partial response text to extract
              features from.
            - ``verifier_outcome``: ``"violated"``, ``"supported"``, or
              ``"abstain"``.  ``"violated"`` is treated as label=1 (high
              risk); ``"supported"`` as label=0 (low risk);  ``"abstain"``
              rows are skipped.
            - ``domain``: Optional domain hint forwarded to feature extraction.
            - ``confidence``: Optional prior confidence hint from the corpus.

            Update rule: one pass of mini-batch-free logistic gradient descent
            (SGD) over all non-abstain rows.  For each row:

                p = sigmoid(w @ x + b)
                error = p - y            # y ∈ {0, 1}
                w -= lr * error * x
                b -= lr * error

            With ``n_passes=1`` and ``lr=0.05`` this acts as a quick online
            correction rather than full re-training.  The weights stay bounded
            because the logistic error is in (-1, 1).

        Args:
            corpus_rows: List of corpus row dicts from Exp 252.
            n_passes:    Number of full passes through the rows. Default 1.
            lr:          Gradient step size.  Default 0.05.

        Spec: REQ-PRED-002, SCENARIO-PRED-004
        """
        if not corpus_rows:
            return

        # Build (x, y) pairs — skip abstain rows.
        pairs: list[tuple[np.ndarray, float]] = []
        for row in corpus_rows:
            outcome = str(row.get("verifier_outcome") or "").lower()
            if outcome == "abstain":
                continue
            label = 1.0 if outcome == "violated" else 0.0
            partial = str(row.get("partial_response") or "")
            dom = str(row.get("domain") or "") or None
            prior = float(row.get("confidence") or 0.5)
            feats = extract_features(partial, domain=dom, prior_confidence=prior)
            pairs.append((feats.to_array(), label))

        if not pairs:
            return

        # SGD over all pairs for n_passes.
        for _ in range(n_passes):
            for x, y in pairs:
                raw = float(np.dot(self._w, x) + self._b)
                p = float(_sigmoid(raw))
                error = p - y
                self._w = self._w - lr * error * x
                self._b = self._b - np.float32(lr * error)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save gate weights to a safetensors file.

        **Detailed explanation for engineers:**
            Two tensors are stored: ``w`` (shape ``(FEATURE_DIM,)``, float32)
            and ``b`` (shape ``(1,)``, float32).  The file is compatible with
            ``load()`` and with any safetensors reader that can handle float32
            tensors.

        Args:
            path: File path to write.  Parent directory must exist.

        Spec: REQ-PRED-003
        """
        tensors = {
            "w": np.array(self._w, dtype=np.float32),
            "b": np.array([float(self._b)], dtype=np.float32),
        }
        save_file(tensors, str(path))

    def load(self, path: str | Path) -> None:
        """Load gate weights from a safetensors file (in-place).

        **Detailed explanation for engineers:**
            Replaces ``self._w`` and ``self._b`` with tensors from the file.
            Validates that both keys are present and that ``w`` has the
            expected ``(FEATURE_DIM,)`` shape before applying.

        Args:
            path: Path to a ``.safetensors`` file written by ``save()``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is missing ``w`` or ``b`` keys, or if
                ``w`` has the wrong shape.

        Spec: REQ-PRED-003
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No safetensors file at: {path}")
        raw = load_file(str(path))
        if "w" not in raw or "b" not in raw:
            missing = [k for k in ("w", "b") if k not in raw]
            raise ValueError(f"safetensors file missing keys: {missing}")
        w = np.array(raw["w"], dtype=np.float32)
        if w.shape != (FEATURE_DIM,):
            raise ValueError(
                f"Weight vector shape mismatch: expected ({FEATURE_DIM},), got {w.shape}"
            )
        self._w = w
        self._b = np.float32(float(raw["b"].ravel()[0]))

    def export_onnx(self, path: str | Path) -> None:
        """Export the gate as a minimal ONNX model.

        **Detailed explanation for engineers:**
            Builds an ONNX graph with a single computation:

                output = Sigmoid(Gemv(w, input) + b)

            where ``input`` is a float32 tensor of shape ``(1, FEATURE_DIM)``
            and ``output`` is a float32 scalar (shape ``(1,)``).

            The graph uses three ONNX operators:
            - ``Reshape`` to ensure the input has shape (1, FEATURE_DIM)
            - ``MatMul`` for the w @ x product
            - ``Add`` for the bias
            - ``Sigmoid`` for the probability output

            The exported model can be run by any ONNX Runtime 1.x+ or an
            NPU runtime (such as the AMD XDNA backend in
            ``carnot.samplers.npu_backend``) without Python or JAX.

        Args:
            path: File path for the ``.onnx`` output.  Parent must exist.

        Raises:
            ImportError: If the ``onnx`` package is not installed.  The error
                message includes the installation hint ``pip install onnx``.

        Spec: REQ-PRED-003
        """
        try:
            import onnx
            from onnx import TensorProto, helper, numpy_helper
        except (ImportError, TypeError) as exc:
            raise ImportError(
                "The 'onnx' package is required for ONNX export. "
                "Install it with: pip install onnx"
            ) from exc

        # Initialiser tensors (constant weights embedded in the graph).
        w_tensor = numpy_helper.from_array(
            self._w.reshape(FEATURE_DIM, 1).astype(np.float32), name="gate_w"
        )
        b_tensor = numpy_helper.from_array(
            np.array([float(self._b)], dtype=np.float32), name="gate_b"
        )

        # Node definitions: input → MatMul(w) → Add(b) → Sigmoid → output
        matmul_node = helper.make_node(
            "MatMul",
            inputs=["input", "gate_w"],
            outputs=["matmul_out"],
        )
        add_node = helper.make_node(
            "Add",
            inputs=["matmul_out", "gate_b"],
            outputs=["logit"],
        )
        sigmoid_node = helper.make_node(
            "Sigmoid",
            inputs=["logit"],
            outputs=["output"],
        )

        # Graph I/O shapes: input (1, FEATURE_DIM) → output (1,)
        input_info = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, FEATURE_DIM]
        )
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1]
        )

        graph = helper.make_graph(
            nodes=[matmul_node, add_node, sigmoid_node],
            name="predictive_gate",
            inputs=[input_info],
            outputs=[output_info],
            initializer=[w_tensor, b_tensor],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
