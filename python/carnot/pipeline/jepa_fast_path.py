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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


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
            self._session = ort.InferenceSession(
                str(self.onnx_path), sess_options=opts
            )
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
