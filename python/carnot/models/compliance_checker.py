"""ComplianceEnergyChecker — KAN-based compliance violation detector.

**Researcher summary (Exp 434):**
    Compliance constraints are STRUCTURAL. "This response gives specific investment
    advice" is a structural claim about language patterns (imperative voice, specific
    numbers, guarantee language), not arithmetic. KAN spline activations can learn to
    assign high energy to responses with compliance-violating patterns the same way
    Safety KAN (Exp 416) learns to assign high energy to jailbreak patterns.

**Why low energy = compliant:**
    Compliant responses in regulated industries share a structural signature: they use
    hedged, advisory language ("may help", "historically has returned", "consult your
    doctor"). Compliance violations have the opposite signature: imperative construction
    ("you must take", "guaranteed to return", "I recommend you buy"), specific quantities
    ("500mg", "20% guaranteed"), and unqualified outcome claims. KAN splines learn these
    structural patterns from labeled examples and encode them as energy contributions.
    A compliant response has smooth, hedged language that activates few high-energy
    spline regions, yielding a low total energy. A violation activates many spline
    regions simultaneously (multiple imperative keywords co-occurring), yielding high
    total energy.

**Why KAN spline inspection is auditable:**
    Unlike black-box classifiers, KAN splines map each human-interpretable keyword
    feature through a learnable 1D function. The control points of the spline for
    feature i (e.g., "buy" count) at hidden unit k directly reveal how that keyword
    affects energy. A compliance auditor can read: "the model learned that 'guarantee'
    raises energy by 2.3 units — this explains why the flagged response received
    high compliance energy." This auditability is legally critical for regulated
    industries. Arbitration, FDA audit, or SEC review can inspect the spline weights
    to verify the model's reasoning. No neural network attention heatmap or SHAP
    approximation is required.

**Phase 3 path:**
    In Phase 3 (EBT foundation model), the compliance layer is a learned energy term
    added to the base EBT energy. Compliance-sensitive inference samples from the
    joint distribution p(x) ∝ exp(-E_base(x) - λ * E_compliance(x)). λ controls the
    compliance penalty strength. Regulated-industry deployments set λ = ∞ (hard
    enforcement). Research deployments set λ = 1 (soft guidance). The KAN compliance
    checker trained here becomes that compliance energy term, serving regulated
    industries that cannot use black-box AI (finance, healthcare, legal services).

Spec: REQ-SAFE-004, REQ-SAFE-005, REQ-SAFE-006
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

# Type alias for the four supported compliance domains.
ComplianceDomain = Literal["financial", "medical", "legal", "general"]

# Domain-specific keyword vocabularies.  Each word/phrase encodes a structural
# pattern that appears in compliance violations for that domain.  The choice of
# keywords is grounded in regulatory language:
#   financial: SEC / FINRA guidance on unauthorized investment advice
#   medical:   FDA guidance on unauthorized practice of medicine
#   legal:     bar association guidance on unauthorized practice of law
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "financial": [
        "buy",
        "sell",
        "invest",
        "guarantee",
        "return",
        "profit",
        "recommend",
        "should invest",
        "will make",
        "certain gain",
        "assured",
        "no risk",
        "double your",
        "triple your",
        "must buy",
        "get rich",
    ],
    "medical": [
        "take",
        "dose",
        "mg",
        "prescribe",
        "diagnose",
        "treat",
        "cure",
        "you should take",
        "you must take",
        "administer",
        "inject",
        "apply",
        "stop taking",
        "discontinue",
        "overdose",
        "medication",
    ],
    "legal": [
        "legally",
        "binding",
        "contract",
        "guarantee",
        "indemnify",
        "liability",
        "obligated",
        "required by law",
        "you must",
        "legally required",
        "sign this",
        "waive",
        "release from",
        "hold harmless",
        "no recourse",
        "irrevocable",
    ],
}

# General domain = union of all domain keywords (for cross-domain screening).
_DOMAIN_KEYWORDS["general"] = list(
    dict.fromkeys(
        kw for kws in [
            _DOMAIN_KEYWORDS["financial"],
            _DOMAIN_KEYWORDS["medical"],
            _DOMAIN_KEYWORDS["legal"],
        ]
        for kw in kws
    )
)


@dataclass
class ComplianceExample:
    """A labeled text sample for training/evaluating ComplianceEnergyChecker.

    Fields:
        text:   The raw text to classify.
        domain: Which compliance domain the example belongs to.
        label:  'compliant' = properly hedged language (target: low energy).
                'violation' = regulatory violation pattern (target: high energy).
    """

    text: str
    domain: ComplianceDomain
    label: Literal["compliant", "violation"]


def encode_compliance_text(
    text: str,
    domain: ComplianceDomain,
    max_features: int = 32,
) -> jnp.ndarray:
    """Encode text as a bag-of-words compliance feature vector.

    For each keyword in the domain's vocabulary, count how many times it
    appears in the text (case-insensitive substring match).  Normalize each
    count by (word_count + 1) to produce a frequency in [0, 1].  Pad or
    truncate to max_features.

    Why substring match instead of word-boundary match:
        Phrases like "should invest" or "legally required" span word boundaries.
        Substring matching captures both single words and multi-word phrases
        uniformly.  Case folding ensures "BUY", "Buy", and "buy" all count.

    Why normalize by word count:
        A 1000-word response with one "buy" is less concerning than a 10-word
        response with one "buy".  Dividing by word count turns raw counts into
        a density that is more comparable across response lengths.

    Args:
        text:         Input text to encode.
        domain:       Which domain's keyword list to use.
        max_features: Output vector length.  Extra keywords are dropped;
                      missing keywords are zero-padded.

    Returns:
        JAX array of shape (max_features,) with float32 values in [0, 1].

    Spec: REQ-SAFE-005
    """
    keywords = _DOMAIN_KEYWORDS[domain]
    text_lower = text.lower()
    word_count = max(len(text.split()), 1)

    features: list[float] = []
    for kw in keywords[:max_features]:
        count = text_lower.count(kw)
        features.append(count / word_count)

    # Zero-pad to max_features if domain has fewer keywords than max_features.
    while len(features) < max_features:
        features.append(0.0)

    return jnp.array(features[:max_features], dtype=jnp.float32)


def _bspline_eval_batch(
    x: jnp.ndarray,
    ctrl: jnp.ndarray,
    n_knots: int,
    degree: int,
) -> jnp.ndarray:
    """Evaluate B-splines for a batch of (input, control-point) pairs.

    This is the vectorised inner kernel shared by all KAN layers.  It uses
    linear interpolation between adjacent control points (a degenerate but
    differentiable B-spline) so that JAX autodiff can compute gradients
    through the spline evaluation.

    Args:
        x:       (n,) input values in [-1, 1].
        ctrl:    (n, n_ctrl) control point arrays, one per input.
        n_knots: Number of knot intervals.
        degree:  Spline degree (used only to determine n_ctrl = n_knots + degree).

    Returns:
        (n,) spline output values.
    """
    n_ctrl = n_knots + degree

    # Map inputs from [0, 1] (feature space) or [-1, 1] (hidden space)
    # to control point index space [0, n_ctrl - 1].
    # Inputs are already in [-1, 1] at this call site.
    normalized = (x + 1.0) / 2.0
    scaled = normalized * (n_knots - 1)

    left = jnp.floor(scaled).astype(jnp.int32)
    left = jnp.clip(left, 0, n_ctrl - 2)
    right = left + 1

    t = jnp.clip(scaled - jnp.floor(scaled), 0.0, 1.0)

    # Gather control point values for each sample.
    batch_idx = jnp.arange(x.shape[0])
    left_vals = ctrl[batch_idx, left]
    right_vals = ctrl[batch_idx, right]

    return left_vals + t * (right_vals - left_vals)


def _compliance_energy(
    features: jnp.ndarray,
    edge_ctrl: jnp.ndarray,
    output_ctrl: jnp.ndarray,
    n_knots: int,
    degree: int,
    n_features: int,
    n_hidden: int,
) -> jnp.ndarray:
    """Pure JAX energy function for a two-layer KAN compliance classifier.

    Architecture (two-layer KAN):
        Layer 1: For each hidden unit k, compute h_k = sum_i spline_ki(x_i).
                 This is a KAN "inner function" — each hidden unit aggregates
                 feature contributions through independent learned splines.
        Layer 2: For each hidden unit k, compute e_k = spline_k(tanh(h_k)).
                 Tanh normalises the hidden activations to [-1, 1] for the
                 output spline domain.  Summing e_k yields the total energy.

    Why two layers:
        A single layer (linear combination of splines) cannot capture feature
        interactions (e.g., "buy" AND "guaranteed" co-occurring).  The hidden
        layer allows the model to learn that certain keyword combinations are
        especially high-energy, which is the structural signature of violations.

    Args:
        features:    (n_features,) input keyword frequencies in [0, 1].
        edge_ctrl:   (n_hidden, n_features, n_ctrl) spline control points for
                     layer 1.
        output_ctrl: (n_hidden, n_ctrl) spline control points for layer 2.
        n_knots:     Number of knots per spline.
        degree:      Spline degree.
        n_features:  Number of input features.
        n_hidden:    Number of hidden units.

    Returns:
        Scalar energy value.

    Spec: REQ-SAFE-004
    """
    # Map features from [0, 1] to [-1, 1] for spline domain compatibility.
    x = features * 2.0 - 1.0  # (n_features,)

    def layer1_hidden_unit(ec_k: jnp.ndarray) -> jnp.ndarray:
        # ec_k: (n_features, n_ctrl)
        # Evaluate spline for each feature independently, then sum.
        vals = _bspline_eval_batch(x, ec_k, n_knots, degree)  # (n_features,)
        return jnp.sum(vals)

    # vmap over n_hidden dimension: each row of edge_ctrl is one hidden unit.
    hidden = jax.vmap(layer1_hidden_unit)(edge_ctrl)  # (n_hidden,)

    # Normalise hidden activations to [-1, 1] so the output splines operate
    # in a consistent domain regardless of n_features scale.
    hidden_norm = jnp.tanh(hidden / (n_features + 1e-8))  # (n_hidden,)

    # Layer 2: output energy via per-hidden-unit splines.
    energies = _bspline_eval_batch(
        hidden_norm, output_ctrl, n_knots, degree
    )  # (n_hidden,)

    return jnp.sum(energies)


def _compute_auroc(scores: list[float], labels: list[int]) -> float:
    """Compute AUC-ROC where higher score = predicted positive (violation).

    Uses the Mann-Whitney U statistic: counts concordant (positive, negative)
    pairs where the positive has a strictly higher score than the negative.
    Ties count as 0.5 each.

    Returns 0.5 if there are no positives or no negatives (degenerate).
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

    # Count concordant pairs: for each violation, count compliant examples
    # with strictly lower energy, plus 0.5 for ties.
    sorted_idx = np.argsort(score_arr)
    sorted_labels = label_arr[sorted_idx]

    cum_neg = 0
    auc_num = 0.0
    for lbl in sorted_labels:
        if lbl == 0:
            cum_neg += 1
        else:
            auc_num += cum_neg  # all lower-scored negatives are concordant

    return float(auc_num) / (n_pos * n_neg)


class ComplianceEnergyChecker:
    """KAN-based compliance violation detector for regulated-industry text.

    This class is the Phase 1 compliance implementation: train on labeled
    compliant/violation examples, then use energy as a classifier score.
    Low energy = compliant; high energy = violation.

    Internal architecture:
        Two-layer KAN (Kolmogorov-Arnold Network) where:
        - Layer 1 maps n_features keyword-frequency inputs to n_hidden hidden
          activations through n_hidden × n_features independent splines.
        - Layer 2 maps each hidden activation through a per-unit output spline
          to produce a scalar energy contribution.
        - Total energy = sum of n_hidden output spline values.

    Training objective (contrastive):
        Loss = mean(E(compliant)) - mean(E(violation)) + L2 regularisation.
        Minimising this loss simultaneously pushes compliant energy down and
        violation energy up.  L2 regularisation prevents the energy from
        diverging to ±∞.

    Serialisation:
        Spline control points are saved via safetensors (float32 arrays).
        Hyperparameters are saved in a companion .json file.

    Auditability (REQ-SAFE-006):
        Call inspect_spline(hidden_unit, feature_idx) to retrieve the control
        points for one spline.  Positive control points mean that feature
        raises energy for that hidden unit.  This is the mechanistic
        explanation for why a text was flagged.

    Example usage:
        >>> checker = ComplianceEnergyChecker(domain='financial')
        >>> checker.train(examples, n_epochs=200)
        >>> checker.is_compliant("XYZ stock may appreciate; past performance...")
        True
        >>> checker.energy("Buy XYZ now! Guaranteed 20% returns!")
        3.7  # high energy = violation

    Spec: REQ-SAFE-004, REQ-SAFE-005, REQ-SAFE-006
    """

    # Fixed spline hyperparameters — not exposed in constructor because they
    # don't need tuning for compliance checking at this feature scale.
    _N_KNOTS: int = 10
    _DEGREE: int = 3

    def __init__(
        self,
        domain: ComplianceDomain,
        n_features: int = 32,
        n_hidden: int = 8,
    ) -> None:
        """Initialise a ComplianceEnergyChecker with random spline weights.

        Args:
            domain:     Which compliance domain to check.
            n_features: Number of bag-of-words features extracted from text.
                        Should match max_features in encode_compliance_text().
            n_hidden:   Number of hidden units in the KAN layer 1.
                        More hidden units = more capacity to model feature
                        interactions, at the cost of more parameters.
        """
        self.domain = domain
        self.n_features = n_features
        self.n_hidden = n_hidden
        self._n_ctrl = self._N_KNOTS + self._DEGREE

        rng = np.random.default_rng(42)
        # edge_ctrl[k, i, :] = spline control points for hidden unit k, input i.
        # Small initialisation so initial energy values are near zero (neutral).
        self.edge_ctrl: np.ndarray = rng.uniform(
            -0.1, 0.1, (n_hidden, n_features, self._n_ctrl)
        ).astype(np.float32)
        # output_ctrl[k, :] = spline control points for output from hidden unit k.
        self.output_ctrl: np.ndarray = rng.uniform(
            -0.1, 0.1, (n_hidden, self._n_ctrl)
        ).astype(np.float32)

    def _energy_from_features(
        self,
        features: jnp.ndarray,
        edge_ctrl: jnp.ndarray | None = None,
        output_ctrl: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Compute energy from a pre-encoded feature vector.

        Delegates to the pure JAX function _compliance_energy so that this
        call site can be used both for inference (with self.edge_ctrl) and for
        gradient computation during training (with arbitrary parameter arrays).
        """
        ec = jnp.array(self.edge_ctrl) if edge_ctrl is None else edge_ctrl
        oc = jnp.array(self.output_ctrl) if output_ctrl is None else output_ctrl
        return _compliance_energy(
            features,
            ec,
            oc,
            self._N_KNOTS,
            self._DEGREE,
            self.n_features,
            self.n_hidden,
        )

    def energy(self, text: str) -> float:
        """Compute compliance energy for a text string.

        Low energy: text matches the structural pattern of compliant language.
        High energy: text matches the structural pattern of a violation.

        Args:
            text: Raw text to evaluate.

        Returns:
            Float energy value.  Sign and magnitude depend on training.

        Spec: REQ-SAFE-004
        """
        features = encode_compliance_text(text, self.domain, self.n_features)
        return float(self._energy_from_features(features))

    def is_compliant(self, text: str, threshold: float = 1.0) -> bool:
        """Return True if the text's compliance energy is below threshold.

        The default threshold of 1.0 is a post-training heuristic.  For
        production use, calibrate the threshold on a held-out validation set
        to achieve the desired false-positive rate.

        Args:
            text:      Raw text to evaluate.
            threshold: Energy value below which the text is considered compliant.

        Returns:
            True if energy < threshold (compliant), False otherwise (violation).

        Spec: REQ-SAFE-004
        """
        return self.energy(text) < threshold

    def train(
        self,
        examples: list[ComplianceExample],
        n_epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        """Train spline weights using contrastive energy minimisation.

        Objective:
            Loss = mean(E(compliant)) - mean(E(violation)) + λ * ||params||²

        Minimising this simultaneously:
        - Pushes compliant example energy DOWN (first term).
        - Pushes violation example energy UP (second term, via negation).
        - Keeps weights small so energy doesn't diverge (regularisation).

        If there are no compliant examples or no violation examples, training
        cannot proceed (contrastive loss requires both classes) and returns
        immediately without updating weights.

        Uses Adam optimiser (Kingma & Ba 2014) via optax for adaptive learning
        rates, which is more robust than SGD on small datasets.

        Args:
            examples: List of ComplianceExample with text, domain, and label.
            n_epochs: Number of full gradient descent steps.
            lr:       Adam learning rate.

        Spec: REQ-SAFE-004
        """
        viol_feats = [
            encode_compliance_text(ex.text, ex.domain, self.n_features)
            for ex in examples
            if ex.label == "violation"
        ]
        comp_feats = [
            encode_compliance_text(ex.text, ex.domain, self.n_features)
            for ex in examples
            if ex.label == "compliant"
        ]

        if not viol_feats or not comp_feats:
            # Contrastive training requires at least one example of each class.
            return

        viol_arr = jnp.stack(viol_feats)  # (n_viol, n_features)
        comp_arr = jnp.stack(comp_feats)  # (n_comp, n_features)

        ec = jnp.array(self.edge_ctrl)
        oc = jnp.array(self.output_ctrl)
        params = (ec, oc)

        def loss_fn(p: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            ec_p, oc_p = p

            def single_energy(f: jnp.ndarray) -> jnp.ndarray:
                return _compliance_energy(
                    f, ec_p, oc_p,
                    self._N_KNOTS, self._DEGREE,
                    self.n_features, self.n_hidden,
                )

            e_comp = jax.vmap(single_energy)(comp_arr)
            e_viol = jax.vmap(single_energy)(viol_arr)

            # Contrastive objective: compliant low, violation high.
            contrastive = jnp.mean(e_comp) - jnp.mean(e_viol)
            # L2 regularisation prevents energy from diverging to -∞.
            reg = 1e-3 * (jnp.sum(ec_p ** 2) + jnp.sum(oc_p ** 2))
            return contrastive + reg

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        grad_fn = jax.jit(jax.value_and_grad(loss_fn))

        for _ in range(n_epochs):
            _loss, grads = grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        self.edge_ctrl = np.array(params[0])
        self.output_ctrl = np.array(params[1])

    def evaluate_auroc(self, examples: list[ComplianceExample]) -> float:
        """Compute AUC-ROC on a labeled example set.

        Uses compliance energy as the classifier score: higher energy →
        predicted violation.  AUC-ROC > 0.5 means the model is doing better
        than random guessing.  AUC-ROC = 1.0 means perfect ranking.

        Args:
            examples: Labeled examples from any domain.

        Returns:
            Float AUC-ROC in [0, 1].  0.5 if no positives or no negatives.

        Spec: REQ-SAFE-004
        """
        scores: list[float] = []
        labels: list[int] = []
        for ex in examples:
            scores.append(self.energy(ex.text))
            labels.append(1 if ex.label == "violation" else 0)
        return _compute_auroc(scores, labels)

    def inspect_spline(self, hidden_unit: int, feature_idx: int) -> np.ndarray:
        """Return spline control points for (hidden_unit, feature_idx).

        This is the primary auditability API (REQ-SAFE-006).  The control
        points reveal what the model learned about a specific keyword's
        contribution to compliance energy.

        Interpretation:
            Control points represent the spline's value at evenly spaced knot
            positions across [-1, 1].  For feature i in the financial domain
            (e.g., feature 0 = "buy" count):
            - Positive control points at high x values: the model learned that
              high "buy" frequency → high energy for this hidden unit → more
              likely to be flagged as a violation.
            - Negative control points: "buy" suppresses energy for this unit.

        Args:
            hidden_unit: Index of the hidden unit (0 to n_hidden - 1).
            feature_idx: Index of the input feature (0 to n_features - 1).

        Returns:
            np.ndarray of shape (n_knots + degree,) = (13,) with defaults.

        Spec: REQ-SAFE-006
        """
        return self.edge_ctrl[hidden_unit, feature_idx].copy()

    def save(self, path: str | Path) -> None:
        """Save spline control points and hyperparameters to disk.

        Writes two files:
        - ``path``: safetensors file with edge_ctrl and output_ctrl arrays.
        - ``path.replace('.safetensors', '.json')``: JSON with hyperparameters.

        Args:
            path: Destination path, should end with ``.safetensors``.

        Spec: REQ-SAFE-004
        """
        from safetensors.numpy import save_file  # local import: optional dep

        tensors = {
            "edge_ctrl": self.edge_ctrl.astype(np.float32),
            "output_ctrl": self.output_ctrl.astype(np.float32),
        }
        save_file(tensors, str(path))

        meta = {
            "domain": self.domain,
            "n_features": self.n_features,
            "n_hidden": self.n_hidden,
            "n_knots": self._N_KNOTS,
            "degree": self._DEGREE,
        }
        meta_path = str(path).rsplit(".safetensors", 1)[0] + ".json"
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ComplianceEnergyChecker":
        """Load a ComplianceEnergyChecker from a saved checkpoint.

        Args:
            path: Path to the .safetensors file (companion .json must exist).

        Returns:
            Fully restored ComplianceEnergyChecker.

        Spec: REQ-SAFE-004
        """
        from safetensors.numpy import load_file  # local import: optional dep

        meta_path = str(path).rsplit(".safetensors", 1)[0] + ".json"
        with open(meta_path) as fh:
            meta = json.load(fh)

        checker = cls(
            domain=meta["domain"],
            n_features=meta["n_features"],
            n_hidden=meta["n_hidden"],
        )
        tensors = load_file(str(path))
        checker.edge_ctrl = np.array(tensors["edge_ctrl"])
        checker.output_ctrl = np.array(tensors["output_ctrl"])
        return checker
