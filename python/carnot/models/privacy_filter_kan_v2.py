"""PrivacyFilterKANv2 — teacher-free PII detection via regex features + KAN energy.

**Researcher summary (Exp 743):**
    Exps 729 and 730 were blocked for two consecutive cycles because the upstream
    dependency `openai/privacy-filter` was unavailable for download.  Two consecutive
    blocked cycles met the governance redesign threshold: the teacher-model dependency
    is retired and replaced with direct feature engineering.

    This v2 trains a two-layer KAN directly on:
    1. Structural PII regex features — credit card (Luhn-valid), SSN (XXX-XX-XXXX),
       email, US phone, IPv4 address, zip code.
    2. Per-pattern statistics — match_count, max_match_length, fraction_matched_chars.
    3. Token statistics — digit_density, alpha_digit_ratio, char_entropy, token_count.
    4. N-gram features — bigram_pii_adj_count (how many adjacent token pairs have
       one PII-matching token).

    No HuggingFace download.  No teacher inference.  No teacher invariant.

**Why contrastive loss without a teacher:**
    With labeled PII and benign corpora (fully synthetic), the contrastive objective
    is `energy(benign) < energy(pii)`.  This is exactly what the teacher-distillation
    pipeline in Exp 729 was trying to approximate using soft teacher labels.  Skipping
    the teacher and using hard corpus labels produces a cleaner training signal when
    the regex features are already high-precision for the PII categories.

**Architecture (same two-layer KAN as PrivacyFilterEnergyChecker v1):**
    - n_features = 23 (6 patterns × 3 stats + 4 token stats + 1 n-gram)
    - n_hidden   = 32
    - n_knots    = 3, degree = 3  (n_ctrl = 6 control points per spline)
    - Layer 1: 32 × 23 splines → 32 hidden activations
    - Layer 2: 32 output splines → sum = E(text)
    - Low energy = benign (no PII).  High energy = privacy violation.

**Gate (REQ-SAFE-020):**
    - AUROC >= 0.80 AND min_tp >= 1 per dataset: publication-ready.
    - AUROC >= 0.85 AND min_tp >= 1 per dataset: supersedes failed v1 target.

Spec: REQ-SAFE-019, REQ-SAFE-020
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax


# ---------------------------------------------------------------------------
# Luhn algorithm — validates credit card numbers
# ---------------------------------------------------------------------------

def luhn_valid(digits: str) -> bool:
    """Return True if the digit string passes the Luhn checksum.

    The Luhn algorithm is used by all major card networks (Visa, Mastercard,
    Amex, Discover) to distinguish valid card numbers from typos.  Implementing
    it here means the CC feature fires ONLY on plausible card numbers — not on
    arbitrary 16-digit strings that happen to appear in benign text.

    Why this matters for false-positive rate:
        Without Luhn validation, any 16-digit sequence (e.g., a benchmark score
        "1234567890123456") would trigger the CC feature.  Luhn cuts FP rate
        dramatically because the probability that a random 16-digit string is
        Luhn-valid is exactly 10% (one valid check digit per prefix).

    Args:
        digits: String of decimal digits (spaces/hyphens already stripped).

    Returns:
        True if the Luhn checksum passes.

    Spec: REQ-SAFE-019
    """
    if not digits.isdigit():
        return False
    total = 0
    reverse = digits[::-1]
    for i, ch in enumerate(reverse):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def luhn_complete(prefix: str) -> str:
    """Append the Luhn check digit to a 15-digit prefix, returning a 16-digit string.

    Used to generate Luhn-valid synthetic credit card numbers for corpus construction.
    Given any 15-digit string, there is exactly one check digit (0-9) that makes
    the 16-digit string Luhn-valid.

    Args:
        prefix: 15-digit decimal string.

    Returns:
        16-digit Luhn-valid string.

    Spec: REQ-SAFE-019
    """
    for check in range(10):
        candidate = prefix + str(check)
        if luhn_valid(candidate):
            return candidate
    raise ValueError(f"No Luhn check digit found for prefix {prefix!r}")


# ---------------------------------------------------------------------------
# Compiled regex patterns (Luhn-aware CC, SSN, email, phone, IP, zip)
# ---------------------------------------------------------------------------

# Credit card: four groups of 4 digits, separator = space or hyphen.
# Luhn validation is applied AFTER regex match in the feature extractor.
_RE_CC = re.compile(r"\b(\d{4})[\s\-](\d{4})[\s\-](\d{4})[\s\-](\d{4})\b")

# SSN: exactly XXX-XX-XXXX where X is a digit.
_RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Email: simplified RFC 5321 subset.
_RE_EMAIL = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")

# US phone: common formats — (NXX) NXX-XXXX, NXX-NXX-XXXX, etc.
_RE_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[\s\-.])?(?:\(\d{3}\)|\d{3})[\s\-.]?\d{3}[\s\-.]?\d{4}(?!\d)"
)

# IPv4: four octets 0-255.
_RE_IP = re.compile(
    r"\b(?:25[0-5]|2\d{2}|1\d{2}|[1-9]\d|\d)"
    r"(?:\.(?:25[0-5]|2\d{2}|1\d{2}|[1-9]\d|\d)){3}\b"
)

# US ZIP code: 5 digits or 5+4 format (ZIP+4).
_RE_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")

# Ordered list of all patterns for iteration.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("cc", _RE_CC),
    ("ssn", _RE_SSN),
    ("email", _RE_EMAIL),
    ("phone", _RE_PHONE),
    ("ip", _RE_IP),
    ("zip", _RE_ZIP),
]

# Total features = 6 patterns × 3 stats + 4 token stats + 1 n-gram = 23.
N_FEATURES_V2: int = 23


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class PrivacyFilterFeatureExtractor:
    """Extract a fixed-size feature vector from text for PII detection.

    This extractor is the core of the v2 teacher-free approach.  Instead of
    running a transformer model to get soft PII labels, it extracts interpretable
    numeric features that directly measure the presence of PII patterns in text.

    Feature layout (23 features total):
        0-2:   CC pattern (Luhn-valid matches): count, max_len, frac_chars
        3-5:   SSN pattern: count, max_len, frac_chars
        6-8:   Email pattern: count, max_len, frac_chars
        9-11:  Phone pattern: count, max_len, frac_chars
        12-14: IP pattern: count, max_len, frac_chars
        15-17: ZIP pattern: count, max_len, frac_chars
        18:    digit_density (digits / total chars)
        19:    alpha_digit_ratio (digits / (alphas + digits + 1))
        20:    char_entropy (Shannon entropy of char distribution, normalised by log2(95))
        21:    token_count (word count / 100, capped at 1.0)
        22:    bigram_pii_adj_count (bigram pairs where one token is PII-like, normalised)

    Why three stats per pattern (not just count):
        `match_count` alone misses intensity.  A single 16-digit CC number and
        ten 16-digit CC numbers both have very different risk levels.
        `max_match_length` captures the longest match (important for distinguishing
        full card numbers from partial matches).  `fraction_matched_chars` captures
        what fraction of the document is covered by PII — a single email in 1000
        words is low-risk; the same email in 3 words is high-risk.

    Spec: REQ-SAFE-019
    """

    def extract(self, text: str) -> np.ndarray:
        """Encode text as a (23,) float32 PII feature vector.

        Pure function: same text → same vector every time (no randomness).
        All features are in [0, 1] so the KAN spline domain [-1, 1] is used
        with a simple linear shift (x * 2 - 1).

        Args:
            text: Raw text to encode (any length).

        Returns:
            np.ndarray of shape (23,) dtype float32.

        Spec: REQ-SAFE-019
        """
        char_count = max(len(text), 1)
        tokens = text.split()
        word_count = max(len(tokens), 1)

        features: list[float] = []

        # Per-pattern features (3 per pattern × 6 patterns = 18 features).
        for name, pattern in _PATTERNS:
            matches = pattern.findall(text)

            # For CC, apply Luhn validation to filter out non-valid card numbers.
            if name == "cc":
                valid_matches = []
                for m in matches:
                    # findall returns tuples of groups for patterns with groups.
                    digits = "".join(m) if isinstance(m, tuple) else m
                    digits_only = re.sub(r"[\s\-]", "", digits)
                    if luhn_valid(digits_only):
                        valid_matches.append(m)
                matches = valid_matches

            raw_spans = pattern.finditer(text)
            span_list = [s.span() for s in raw_spans]

            if name == "cc" and not matches:
                # All CC matches were Luhn-invalid; no spans.
                span_list = []

            match_count = len(matches) / word_count
            max_len = max((e - s for s, e in span_list), default=0) / char_count
            frac_chars = sum(e - s for s, e in span_list) / char_count

            features.extend([
                min(match_count, 1.0),
                min(max_len, 1.0),
                min(frac_chars, 1.0),
            ])

        # Token statistics (features 18-21).
        chars = list(text)
        digits = [c for c in chars if c.isdigit()]
        alphas = [c for c in chars if c.isalpha()]

        digit_density = len(digits) / char_count
        alpha_digit_ratio = len(digits) / (len(alphas) + len(digits) + 1)

        # Shannon entropy of the character distribution, normalised by log2(95)
        # (95 printable ASCII characters).  High entropy text (mixed alphanumeric
        # + symbols) is more PII-like than pure prose.
        from collections import Counter
        freq = Counter(chars)
        entropy = 0.0
        for cnt in freq.values():
            p = cnt / char_count
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(95)  # upper bound for printable ASCII
        char_entropy = min(entropy / max_entropy, 1.0)

        token_count_norm = min(word_count / 100.0, 1.0)

        features.extend([digit_density, alpha_digit_ratio, char_entropy, token_count_norm])

        # N-gram feature (feature 22): bigrams where one token matches any PII pattern.
        # A token is "PII-like" if at least one compiled pattern finds a match in it.
        bigram_pii_adj = 0
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i + 1]
            t1_pii = any(p.search(t1) for _, p in _PATTERNS)
            t2_pii = any(p.search(t2) for _, p in _PATTERNS)
            if t1_pii or t2_pii:
                bigram_pii_adj += 1
        bigram_norm = bigram_pii_adj / max(word_count - 1, 1)
        features.append(min(bigram_norm, 1.0))

        assert len(features) == N_FEATURES_V2, f"Expected {N_FEATURES_V2}, got {len(features)}"
        return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# KAN energy function (JAX pure, identical structure to v1 but parameterised)
# ---------------------------------------------------------------------------

def _bspline_eval_batch(
    x: jnp.ndarray,
    ctrl: jnp.ndarray,
    n_knots: int,
    degree: int,
) -> jnp.ndarray:
    """Piecewise-linear spline evaluation for a batch of (input, control-point) pairs.

    This is the same spline kernel used in PrivacyFilterEnergyChecker v1 and
    PromptInjectionEnergyChecker.  Linear interpolation between adjacent control
    points is differentiable through JAX autodiff and numerically stable in float32.

    Args:
        x:       (n,) inputs in [-1, 1].
        ctrl:    (n, n_ctrl) control point arrays.
        n_knots: Number of knot intervals.
        degree:  Spline degree (n_ctrl = n_knots + degree).

    Returns:
        (n,) spline outputs.
    """
    n_ctrl = n_knots + degree
    normalized = (x + 1.0) / 2.0
    scaled = normalized * (n_knots - 1)
    left = jnp.clip(jnp.floor(scaled).astype(jnp.int32), 0, n_ctrl - 2)
    right = left + 1
    t = jnp.clip(scaled - jnp.floor(scaled), 0.0, 1.0)
    idx = jnp.arange(x.shape[0])
    return ctrl[idx, left] + t * (ctrl[idx, right] - ctrl[idx, left])


def _kan_energy(
    features: jnp.ndarray,
    edge_ctrl: jnp.ndarray,
    output_ctrl: jnp.ndarray,
    n_knots: int,
    degree: int,
    n_features: int,
    n_hidden: int,
) -> jnp.ndarray:
    """Two-layer KAN energy function for PrivacyFilterKANv2.

    Layer 1: hidden_k = sum_i spline_ki(feature_i)   for k in 0..n_hidden-1
    Layer 2: e_k = spline_k(tanh(hidden_k / n_features))
    Energy = sum(e_k)

    Low energy = benign text.  High energy = PII text.

    Args:
        features:    (n_features,) PII feature densities in [0, 1].
        edge_ctrl:   (n_hidden, n_features, n_ctrl) layer-1 control points.
        output_ctrl: (n_hidden, n_ctrl) layer-2 control points.
        n_knots, degree, n_features, n_hidden: architecture constants.

    Returns:
        Scalar energy.
    """
    x = features * 2.0 - 1.0  # map [0, 1] → [-1, 1]

    def layer1_unit(ec_k: jnp.ndarray) -> jnp.ndarray:
        vals = _bspline_eval_batch(x, ec_k, n_knots, degree)
        return jnp.sum(vals)

    hidden = jax.vmap(layer1_unit)(edge_ctrl)                     # (n_hidden,)
    hidden_norm = jnp.tanh(hidden / (n_features + 1e-8))          # (n_hidden,)
    energies = _bspline_eval_batch(hidden_norm, output_ctrl, n_knots, degree)
    return jnp.sum(energies)


# ---------------------------------------------------------------------------
# PrivacyFilterKANv2 — the complete model
# ---------------------------------------------------------------------------

@dataclass
class PrivacyExampleV2:
    """Labeled text sample for PrivacyFilterKANv2 training and evaluation.

    Fields:
        text:   Raw text to classify.
        label:  'benign' = no PII (target: low energy).
                'pii'    = contains PII (target: high energy).
        source: Dataset provenance tag.
    """

    text: str
    label: Literal["benign", "pii"]
    source: str = "unknown"


class PrivacyFilterKANv2:
    """KAN-based PII energy model trained directly on regex features (no teacher).

    This class provides:
    - energy(text) -> float     — scalar PII energy; high = likely PII.
    - is_safe(text, threshold)  — True when energy <= threshold.
    - train(benign, pii)        — contrastive training on PrivacyExampleV2 lists.
    - evaluate_auroc(examples)  — AUC-ROC on labeled examples.
    - save(path) / load(path)   — JSON weight serialisation.

    The model is CPU-only: no GPU required.  Forward pass target: < 5 ms.

    Architecture:
        n_features=23, n_hidden=32, n_knots=3, degree=3 → n_ctrl=6.
        Total parameters: 32×23×6 + 32×6 = 4416 + 192 = 4608.

    Spec: REQ-SAFE-019, REQ-SAFE-020
    """

    _N_KNOTS: int = 3
    _DEGREE: int = 3

    def __init__(self, n_features: int = N_FEATURES_V2, n_hidden: int = 32) -> None:
        """Initialise PrivacyFilterKANv2 with random weights.

        Args:
            n_features: Number of input features (default: N_FEATURES_V2 = 23).
            n_hidden:   Number of hidden KAN units (default: 32).
        """
        self.n_features = n_features
        self.n_hidden = n_hidden
        self._extractor = PrivacyFilterFeatureExtractor()

        n_ctrl = self._N_KNOTS + self._DEGREE
        rng = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(rng)
        # Small-magnitude random initialisation: prevents dead splines at start.
        self._edge_ctrl: jnp.ndarray = jax.random.normal(k1, (n_hidden, n_features, n_ctrl)) * 0.01
        self._output_ctrl: jnp.ndarray = jax.random.normal(k2, (n_hidden, n_ctrl)) * 0.01

    @property
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        return int(self._edge_ctrl.size + self._output_ctrl.size)

    def _energy_from_features(self, features: jnp.ndarray) -> float:
        """Compute energy from a pre-extracted feature vector."""
        return float(
            _kan_energy(
                features,
                self._edge_ctrl,
                self._output_ctrl,
                self._N_KNOTS,
                self._DEGREE,
                self.n_features,
                self.n_hidden,
            )
        )

    def energy(self, text: str) -> float:
        """Compute PII energy for text.  Low = benign, high = PII violation.

        Args:
            text: Raw text to score.

        Returns:
            Scalar float energy.

        Spec: REQ-SAFE-019
        """
        features = jnp.array(self._extractor.extract(text), dtype=jnp.float32)
        return self._energy_from_features(features)

    def is_safe(self, text: str, threshold: float = 0.0) -> bool:
        """Return True if energy(text) <= threshold (no PII detected).

        Args:
            text:      Raw text to check.
            threshold: Energy threshold separating benign from PII.

        Returns:
            True if text is predicted benign (safe).

        Spec: REQ-SAFE-019
        """
        return self.energy(text) <= threshold

    def train(
        self,
        benign: list[PrivacyExampleV2],
        pii: list[PrivacyExampleV2],
        n_epochs: int = 100,
        lr: float = 1e-3,
        margin: float = 1.0,
    ) -> list[float]:
        """Train with contrastive loss: energy(benign) < energy(pii).

        Uses full-batch gradient descent (all examples per epoch).  The margin
        loss pushes benign energy below (pii energy - margin), ensuring a clear
        separation in energy space.

        Args:
            benign:   List of benign PrivacyExampleV2 samples.
            pii:      List of PII PrivacyExampleV2 samples.
            n_epochs: Number of gradient steps.
            lr:       Adam learning rate.
            margin:   Contrastive margin (energy_pii - energy_benign >= margin).

        Returns:
            List of per-epoch loss values.

        Spec: REQ-SAFE-019
        """
        extractor = self._extractor
        benign_arr = jnp.array(
            [extractor.extract(e.text) for e in benign], dtype=jnp.float32
        )
        pii_arr = jnp.array(
            [extractor.extract(e.text) for e in pii], dtype=jnp.float32
        )

        n_knots = self._N_KNOTS
        degree = self._DEGREE
        n_features = self.n_features
        n_hidden = self.n_hidden

        params = (self._edge_ctrl, self._output_ctrl)
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        @jax.jit
        def loss_fn(params, benign_batch, pii_batch):
            ec, oc = params

            def energy_one(feat):
                return _kan_energy(feat, ec, oc, n_knots, degree, n_features, n_hidden)

            e_benign = jax.vmap(energy_one)(benign_batch)  # (n_benign,)
            e_pii = jax.vmap(energy_one)(pii_batch)        # (n_pii,)

            # Mean energies: benign should be low, PII should be high.
            mean_e_b = jnp.mean(e_benign)
            mean_e_p = jnp.mean(e_pii)
            # Hinge loss: max(0, margin - (e_pii - e_benign)).
            return jnp.maximum(0.0, margin - (mean_e_p - mean_e_b))

        @jax.jit
        def step(params, opt_state, benign_batch, pii_batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, benign_batch, pii_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        loss_curve = []
        for _ in range(n_epochs):
            params, opt_state, loss = step(params, opt_state, benign_arr, pii_arr)
            loss_curve.append(float(loss))

        self._edge_ctrl, self._output_ctrl = params
        return loss_curve

    def evaluate_auroc(self, examples: list[PrivacyExampleV2]) -> float:
        """Compute AUC-ROC on a list of labeled PrivacyExampleV2 objects.

        Higher energy = predicted PII (positive class).

        Args:
            examples: Mixed list of 'benign' and 'pii' labeled samples.

        Returns:
            AUROC float in [0, 1].  Returns 0.5 for degenerate label sets.

        Spec: REQ-SAFE-020
        """
        scores = [self.energy(e.text) for e in examples]
        labels = [1 if e.label == "pii" else 0 for e in examples]
        return _compute_auroc(scores, labels)

    def save(self, path: Path | str) -> None:
        """Save weights to a JSON file with schema='carnot.privacy_filter_kan.v2'.

        Args:
            path: Output path (.json extension expected).

        Spec: REQ-SAFE-019
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "carnot.privacy_filter_kan.v2",
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
    def load(cls, path: Path | str) -> "PrivacyFilterKANv2":
        """Load weights from a JSON file saved by save().

        Args:
            path: Path to .json file written by save().

        Returns:
            Loaded PrivacyFilterKANv2 with restored weights.

        Raises:
            ValueError: if schema field doesn't match 'carnot.privacy_filter_kan.v2'.

        Spec: REQ-SAFE-019
        """
        path = Path(path)
        with open(path) as fh:
            payload = json.load(fh)
        if payload.get("schema") != "carnot.privacy_filter_kan.v2":
            raise ValueError(
                f"Unexpected schema: {payload.get('schema')!r}; "
                "expected 'carnot.privacy_filter_kan.v2'"
            )
        model = cls(n_features=payload["n_features"], n_hidden=payload["n_hidden"])
        model._edge_ctrl = jnp.array(payload["edge_ctrl"], dtype=jnp.float32)
        model._output_ctrl = jnp.array(payload["output_ctrl"], dtype=jnp.float32)
        return model


# ---------------------------------------------------------------------------
# AUROC computation (copied from privacy_filter_kan.py for independence)
# ---------------------------------------------------------------------------

def _compute_auroc(scores: list[float], labels: list[int]) -> float:
    """Compute AUC-ROC where higher score = predicted positive (PII).

    Uses the trapezoidal rule on the (FPR, TPR) curve sorted by descending score.
    Returns 0.5 for degenerate label sets (all same class).

    Spec: REQ-SAFE-020
    """
    if len(set(labels)) < 2:
        return 0.5
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        tpr = tp / n_pos
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr, prev_tpr = fpr, tpr
    return auc
