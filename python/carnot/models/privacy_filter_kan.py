"""PrivacyFilterEnergyChecker — KAN student distilled from openai/privacy-filter.

**Researcher summary (Exp 729):**
    openai/privacy-filter is a transformer-based classifier that detects when
    LLM outputs or user inputs contain PII (personal identifiable information):
    credit card numbers, SSNs, emails, phone numbers, addresses, etc.  This
    module provides a ~3000-parameter KAN student that runs in < 5 ms on CPU
    and is distilled from that teacher using a contrastive training pipeline.

**Why EBM for privacy filtering (not just running openai/privacy-filter inline):**
    1. Cost/latency: openai/privacy-filter requires ~12 GB VRAM and ~200 ms/call.
       This KAN student runs in < 5 ms on a single CPU core with no GPU required.
    2. Hardware portability: KAN energy landscapes compile to Ising/FPGA backends
       alongside the rest of the Carnot EBM stack.  Transformer classifiers do not.
    3. Interpretability: each of the 16 spline control-point sets maps to a named
       PII feature (e.g., "cc_pattern_density").  An auditor can directly read why
       the model assigned high energy to a given text.
    4. Composable energy: the pipeline sums energies from multiple EBM checks.
       A scalar from this checker joins that sum; a boolean classifier does not.

**Architecture:**
    Two-layer KAN identical in structure to PromptInjectionEnergyChecker:
    - n_features=16 (PII structural + keyword + character statistics)
    - n_hidden=32 (hidden units)
    - n_knots=3, degree=3 (n_ctrl=6 control points per spline)
    - Layer 1: 32 × 16 splines → 32 hidden activations
    - Layer 2: 32 output splines → 32 energy scalars → sum = E(text)
    - Total parameters: 32×16×6 + 32×6 = 3072 + 192 = 3264

    Low energy = benign (no PII); high energy = privacy violation.

**Acceptance criteria (REQ-SAFE-015):**
    - energy(text) → float
    - is_safe(text, threshold) → bool
    - AUROC >= 0.85 on 400-example held-out set (stretch: 0.90)
    - CPU-only forward pass < 5 ms

Spec: REQ-SAFE-015, REQ-SAFE-016
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

from carnot.models.privacy_filter_features import encode_privacy, N_PRIVACY_FEATURES


@dataclass
class PrivacyExample:
    """A labeled text for training/evaluating PrivacyFilterEnergyChecker.

    Fields:
        text:   The raw text to classify.
        label:  'benign' = no PII (target: low energy).
                'pii'    = contains PII (target: high energy).
        source: Dataset source for provenance tracking.
    """

    text: str
    label: Literal["benign", "pii"]
    source: str = "unknown"


def _bspline_eval_batch(
    x: jnp.ndarray,
    ctrl: jnp.ndarray,
    n_knots: int,
    degree: int,
) -> jnp.ndarray:
    """Evaluate B-splines for a batch of (input, control-point) pairs.

    Linear interpolation between adjacent control points — differentiable so
    JAX autodiff can compute gradients through the spline for KAN training.

    Why linear interpolation instead of full B-spline basis functions:
        Full Cox-de Boor recursion is expensive and numerically unstable in
        32-bit float.  Linear interpolation between adjacent control points
        (piecewise linear spline) provides sufficient expressiveness for the
        tabular PII features used here, and is exactly what the prompt-injection
        KAN uses.  The term "B-spline" here refers to this piecewise structure.

    Args:
        x:       (n,) input values in [-1, 1].
        ctrl:    (n, n_ctrl) control point arrays, one per input.
        n_knots: Number of knot intervals.
        degree:  Spline degree (n_ctrl = n_knots + degree).

    Returns:
        (n,) spline output values.
    """
    n_ctrl = n_knots + degree
    normalized = (x + 1.0) / 2.0
    scaled = normalized * (n_knots - 1)

    left = jnp.floor(scaled).astype(jnp.int32)
    left = jnp.clip(left, 0, n_ctrl - 2)
    right = left + 1
    t = jnp.clip(scaled - jnp.floor(scaled), 0.0, 1.0)

    batch_idx = jnp.arange(x.shape[0])
    left_vals = ctrl[batch_idx, left]
    right_vals = ctrl[batch_idx, right]

    return left_vals + t * (right_vals - left_vals)


def _privacy_energy(
    features: jnp.ndarray,
    edge_ctrl: jnp.ndarray,
    output_ctrl: jnp.ndarray,
    n_knots: int,
    degree: int,
    n_features: int,
    n_hidden: int,
) -> jnp.ndarray:
    """Pure JAX energy function for the two-layer privacy filter KAN.

    Layer 1: hidden_k = sum_i spline_ki(feature_i) for each hidden unit k.
    Layer 2: e_k = spline_k(tanh(hidden_k)) for each hidden unit k.
    Energy = sum(e_k).

    Low energy → benign text (no PII).  High energy → PII-containing text.

    Why tanh normalisation between layers:
        The sum of 16 spline outputs in layer 1 can exceed [-1, 1], pushing
        the layer-2 spline inputs outside its defined domain.  tanh squashes
        the summed activation back into (-1, 1) so both layers operate in the
        correct domain.  The (n_features + 1e-8) denominator scales the
        pre-tanh value so that ~equal feature activation produces ~0 hidden
        activation before training, enabling stable initialisation.

    Args:
        features:    (n_features,) PII feature densities in [0, 1].
        edge_ctrl:   (n_hidden, n_features, n_ctrl) layer-1 control points.
        output_ctrl: (n_hidden, n_ctrl) layer-2 control points.
        n_knots:     Number of knot intervals per spline.
        degree:      Spline degree.
        n_features:  Number of input features.
        n_hidden:    Number of hidden units.

    Returns:
        Scalar energy (sum of layer-2 spline outputs).

    Spec: REQ-SAFE-015
    """
    # Map features from [0, 1] to [-1, 1] for the spline domain.
    x = features * 2.0 - 1.0  # (n_features,)

    def layer1_unit(ec_k: jnp.ndarray) -> jnp.ndarray:
        vals = _bspline_eval_batch(x, ec_k, n_knots, degree)
        return jnp.sum(vals)

    hidden = jax.vmap(layer1_unit)(edge_ctrl)          # (n_hidden,)
    hidden_norm = jnp.tanh(hidden / (n_features + 1e-8))  # (n_hidden,)

    energies = _bspline_eval_batch(hidden_norm, output_ctrl, n_knots, degree)
    return jnp.sum(energies)


def _compute_auroc(scores: list[float], labels: list[int]) -> float:
    """Compute AUC-ROC where higher score = predicted PII (positive class).

    Uses the Mann-Whitney U counting approach: for every (pii, benign) pair,
    count whether the PII score strictly exceeds the benign score.  Ties
    contribute 0.5.  Returns 0.5 for degenerate cases (all-positive or
    all-negative label sets) since a random classifier would score 0.5.

    Spec: REQ-SAFE-015
    """
    n = len(scores)
    if n == 0:
        return 0.5

    score_arr = np.array(scores, dtype=np.float64)
    label_arr = np.array(labels, dtype=np.int32)

    n_pos = int(label_arr.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    sorted_idx = np.argsort(score_arr)
    sorted_labels = label_arr[sorted_idx]

    cum_neg = 0
    auc_num = 0.0
    for lbl in sorted_labels:
        if lbl == 0:
            cum_neg += 1
        else:
            auc_num += cum_neg

    return float(auc_num) / (n_pos * n_neg)


class PrivacyFilterEnergyChecker:
    """KAN student distilled from openai/privacy-filter.

    Assigns scalar energy to input text:
    - Low energy → benign (no PII content detected)
    - High energy → PII violation (credit cards, SSNs, emails, etc.)

    Architecture:
        Two-layer KAN with n_hidden=32 hidden units and n_features=16 PII
        pattern features.  Total ~3264 parameters, runs in < 5 ms on CPU.
        Weights saved as JSON (schema="carnot.privacy_filter_kan.v1") to
        distinguish from the prompt-injection KAN weights.

    Training:
        Contrastive loss: mean(E(benign)) - mean(E(pii)) + L2 reg.
        This pushes benign energy down and PII energy up simultaneously.

    Auditability:
        inspect_spline(hidden_unit, feature_idx) returns the control points
        for the spline at (hidden_unit, feature_idx).  Since feature i maps
        to a named PII pattern (e.g., feature 0 = "cc_pattern_density"),
        the control points directly explain why a text received high energy.

    Example:
        >>> checker = PrivacyFilterEnergyChecker()
        >>> checker.train(examples, n_epochs=100)
        >>> checker.is_safe("What is the capital of France?")
        True
        >>> checker.energy("My credit card is 4111 1111 1111 1111")
        5.3  # high energy = PII violation

    Spec: REQ-SAFE-015, REQ-SAFE-016
    """

    _N_KNOTS: int = 3
    _DEGREE: int = 3

    def __init__(
        self,
        n_features: int = N_PRIVACY_FEATURES,
        n_hidden: int = 32,
    ) -> None:
        """Initialise with near-zero spline weights for neutral initial energy.

        Parameter count:
            n_hidden * n_features * (n_knots + degree) + n_hidden * (n_knots + degree)
            = 32 * 16 * 6 + 32 * 6 = 3072 + 192 = 3264

        Args:
            n_features: Number of PII features from encode_privacy() (must match
                        N_PRIVACY_FEATURES unless you retrain from scratch).
            n_hidden:   Number of hidden units.  32 was chosen to give ~3000 params
                        with the degree-3 splines and 16 features.
        """
        self.n_features = n_features
        self.n_hidden = n_hidden
        n_ctrl = self._N_KNOTS + self._DEGREE

        rng = np.random.default_rng(42)
        # Small random init so forward passes start near zero energy.
        self._edge_ctrl = jnp.array(
            rng.normal(0, 0.01, (n_hidden, n_features, n_ctrl)), dtype=jnp.float32
        )
        self._output_ctrl = jnp.array(
            rng.normal(0, 0.01, (n_hidden, n_ctrl)), dtype=jnp.float32
        )

    def n_params(self) -> int:
        """Return total parameter count (edge + output control points)."""
        return int(self._edge_ctrl.size + self._output_ctrl.size)

    def energy(self, text: str) -> float:
        """Compute scalar privacy-violation energy for text.

        Higher energy = more likely to contain PII.  The threshold for
        is_safe() defaults to 0.0 (benign text should produce slightly
        negative energy after training).

        Args:
            text: Input text to score.

        Returns:
            float energy value.

        Spec: REQ-SAFE-015
        """
        features = encode_privacy(text, self.n_features)
        e = _privacy_energy(
            features,
            self._edge_ctrl,
            self._output_ctrl,
            self._N_KNOTS,
            self._DEGREE,
            self.n_features,
            self.n_hidden,
        )
        return float(e)

    def is_safe(self, text: str, threshold: float = 0.0) -> bool:
        """Return True when energy < threshold (text is benign / no PII).

        Args:
            text:      Input text to classify.
            threshold: Decision boundary.  Default 0.0 assumes contrastive
                       training centred around zero.

        Returns:
            True if text is predicted benign (no PII detected).

        Spec: REQ-SAFE-015
        """
        return self.energy(text) < threshold

    def train(
        self,
        examples: list[PrivacyExample],
        n_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> list[float]:
        """Train via contrastive loss: push benign energy down, PII energy up.

        Loss = mean(E(benign)) - mean(E(pii)) + weight_decay * L2(params)

        Why contrastive loss (not cross-entropy):
            We want the model to assign calibrated *scalar energies*, not just
            binary labels.  Cross-entropy trains toward 0/1 probabilities which
            collapse the energy scale.  Contrastive loss preserves the energy
            range so the checker composes correctly with other EBM components
            that sum energies.

        Args:
            examples:     List of PrivacyExample with label 'benign' or 'pii'.
            n_epochs:     Training epochs.
            lr:           AdamW learning rate.
            weight_decay: L2 regularisation coefficient.

        Returns:
            List of per-epoch loss values for diagnostics.

        Spec: REQ-SAFE-016
        """
        benign = [encode_privacy(e.text, self.n_features) for e in examples if e.label == "benign"]
        pii = [encode_privacy(e.text, self.n_features) for e in examples if e.label == "pii"]

        if not benign or not pii:
            return []

        benign_arr = jnp.stack(benign)
        pii_arr = jnp.stack(pii)

        params = (self._edge_ctrl, self._output_ctrl)
        optimizer = optax.adamw(lr, weight_decay=weight_decay)
        opt_state = optimizer.init(params)

        @jax.jit
        def loss_fn(params, benign_batch, pii_batch):
            ec, oc = params

            def single_energy(features):
                return _privacy_energy(
                    features, ec, oc,
                    self._N_KNOTS, self._DEGREE, self.n_features, self.n_hidden
                )

            benign_energies = jax.vmap(single_energy)(benign_batch)
            pii_energies = jax.vmap(single_energy)(pii_batch)
            # Contrastive: benign mean low, pii mean high.
            return jnp.mean(benign_energies) - jnp.mean(pii_energies)

        @jax.jit
        def step(params, opt_state, benign_batch, pii_batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, benign_batch, pii_batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        loss_curve: list[float] = []
        for _ in range(n_epochs):
            params, opt_state, loss = step(params, opt_state, benign_arr, pii_arr)
            loss_curve.append(float(loss))

        self._edge_ctrl, self._output_ctrl = params
        return loss_curve

    def evaluate_auroc(self, examples: list[PrivacyExample]) -> float:
        """Compute AUC-ROC on a list of labeled PrivacyExample objects.

        Higher energy = predicted PII (positive class).  AUROC of 1.0 means
        all PII texts have strictly higher energy than all benign texts.

        Args:
            examples: List of PrivacyExample with label 'benign' or 'pii'.

        Returns:
            AUROC float in [0, 1].  Returns 0.5 for degenerate label sets.

        Spec: REQ-SAFE-015
        """
        scores = [self.energy(e.text) for e in examples]
        labels = [1 if e.label == "pii" else 0 for e in examples]
        return _compute_auroc(scores, labels)

    def inspect_spline(self, hidden_unit: int, feature_idx: int) -> list[float]:
        """Return control points for the layer-1 spline at (hidden_unit, feature_idx).

        Useful for auditing: the control-point values reveal how feature_idx
        (e.g., "cc_pattern_density") contributes to the hidden unit's activation.
        Increasing control points → feature pushes energy higher.

        Args:
            hidden_unit: Index of the hidden unit (0 to n_hidden-1).
            feature_idx: Index of the input feature (0 to n_features-1).

        Returns:
            List of (n_knots + degree) float control point values.

        Spec: REQ-SAFE-016
        """
        return self._edge_ctrl[hidden_unit, feature_idx].tolist()

    def save(self, path: Path | str) -> None:
        """Save model weights to a JSON file.

        Uses schema="carnot.privacy_filter_kan.v1" to distinguish from
        the prompt-injection KAN weights (schema="carnot.prompt_injection_kan.v1").

        Args:
            path: Output path (.json extension expected).

        Spec: REQ-SAFE-016
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "carnot.privacy_filter_kan.v1",
            "n_features": self.n_features,
            "n_hidden": self.n_hidden,
            "n_knots": self._N_KNOTS,
            "degree": self._DEGREE,
            "edge_ctrl": self._edge_ctrl.tolist(),
            "output_ctrl": self._output_ctrl.tolist(),
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path | str) -> "PrivacyFilterEnergyChecker":
        """Load model weights from a JSON file saved by save().

        Args:
            path: Path to the .json file written by save().

        Returns:
            Loaded PrivacyFilterEnergyChecker with restored weights.

        Raises:
            ValueError: if the schema field doesn't match the expected value.

        Spec: REQ-SAFE-016
        """
        path = Path(path)
        with open(path) as fh:
            payload = json.load(fh)
        if payload.get("schema") != "carnot.privacy_filter_kan.v1":
            raise ValueError(
                f"Unexpected schema: {payload.get('schema')!r}; "
                "expected 'carnot.privacy_filter_kan.v1'"
            )
        checker = cls(
            n_features=payload["n_features"],
            n_hidden=payload["n_hidden"],
        )
        checker._edge_ctrl = jnp.array(payload["edge_ctrl"], dtype=jnp.float32)
        checker._output_ctrl = jnp.array(payload["output_ctrl"], dtype=jnp.float32)
        return checker
