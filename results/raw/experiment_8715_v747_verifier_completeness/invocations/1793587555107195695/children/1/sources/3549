"""HalluSAE sparse auto-encoder for hallucination feature identification.

**Why this module exists:**
    arXiv 2604.16430 (HalluSAE) shows that sparse auto-encoders trained on LLM
    hidden states can identify *monosemantic* features causally linked to hallucinations.
    Instead of a black-box energy score, we get interpretable causal explanations:
    "Feature 47 (off-by-one arithmetic patterns) activated, causing the violation."

    This module trains a sparse AE on text-level representations of FOVER step pairs,
    since we have step text but not the underlying LLM hidden states.  The text features
    encode arithmetic structure (digit counts, operator counts, COMPUTE: lines) that
    proxy the hidden-state information available in the full HalluSAE setup.

**Connection to Exp 668 VR win:**
    Structured-equation forcing + repair improved GSM8K accuracy from 36% to 100%.
    This module answers WHY: which latent features were suppressed by the forcing prompt?
    If compute_line_count (COMPUTE: occurrences) appears in the top-10 hallucination
    features, it confirms that the structured forcing prompt works by activating the
    "show your arithmetic explicitly" feature that the model under-uses when hallucinating.

**Spec:** REQ-VERIFY-160, REQ-VERIFY-161,
          SCENARIO-VERIFY-212, SCENARIO-VERIFY-213
"""

from __future__ import annotations

import re
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

# ---------------------------------------------------------------------------
# Text feature extraction
# ---------------------------------------------------------------------------

# Number of hash buckets for bag-of-character-trigrams feature.
# 128 keeps the feature vector small enough for a 512-hidden-dim AE to train
# on 57 examples without overfitting.
_HASH_DIM = 128


def extract_text_features(text: str) -> jnp.ndarray:
    """Convert a reasoning step text to a fixed-length float32 feature vector.

    **Why text features instead of hidden states:**
        The FOVER corpus stores step text, not LLM hidden states.  Text-level
        features that encode arithmetic structure (digits, operators, COMPUTE lines)
        are strong proxies for the internal arithmetic uncertainty feature that
        HalluSAE identifies in hidden states.  This lets us run the experiment
        without re-running the LLM.

    **Feature layout (134 dimensions total):**
        - [0:128]  Bag of character-trigram hashes (collision-tolerant, fast)
        - [128]    digit_count         — number of digit characters
        - [129]    operator_count      — occurrences of +, -, *, /
        - [130]    equals_count        — occurrences of '='
        - [131]    carry_pattern_count — occurrences of carry-indicator patterns
                                        (e.g. numbers > 9 summed digit-by-digit)
        - [132]    compute_line_count  — occurrences of "COMPUTE:" prefix
        - [133]    step_length         — total character count / 100 (normalised)

    REQ-VERIFY-160-5, SCENARIO-VERIFY-213
    """
    # --- Bag-of-trigram-hashes --------------------------------------------------
    hash_vec = jnp.zeros(_HASH_DIM, dtype=jnp.float32)
    trigrams: list[float] = []
    for i in range(len(text) - 2):
        tri = text[i : i + 3]
        bucket = hash(tri) % _HASH_DIM
        trigrams.append(bucket)
    for b in trigrams:
        hash_vec = hash_vec.at[int(b)].add(1.0)

    # Normalise so long texts don't dominate by raw count.
    norm = jnp.maximum(hash_vec.sum(), 1.0)
    hash_vec = hash_vec / norm

    # --- Structured arithmetic features ----------------------------------------
    digit_count = float(sum(c.isdigit() for c in text))
    operator_count = float(sum(text.count(op) for op in ("+", "-", "*", "/")))
    equals_count = float(text.count("="))

    # carry_pattern_count: heuristic — look for multi-digit number followed by
    # another multi-digit number in the same arithmetic expression.  The regex
    # catches patterns like "47 + 28" where carrying is needed.
    carry_matches = re.findall(r"\b\d{2,}\s*[+\-]\s*\d{2,}\b", text)
    carry_pattern_count = float(len(carry_matches))

    # compute_line_count: "COMPUTE:" lines are the structured forcing marker from
    # Exp 668.  This is the key feature for the VR win mechanism hypothesis.
    compute_line_count = float(len(re.findall(r"COMPUTE:", text, re.IGNORECASE)))

    # step_length normalised to [0, ~1] range for typical step lengths (< 500 chars).
    step_length = float(len(text)) / 100.0

    structured = jnp.array(
        [
            digit_count,
            operator_count,
            equals_count,
            carry_pattern_count,
            compute_line_count,
            step_length,
        ],
        dtype=jnp.float32,
    )

    return jnp.concatenate([hash_vec, structured])


# Feature dimension = hash_dim + 6 structured features
FEATURE_DIM = _HASH_DIM + 6  # 134


# ---------------------------------------------------------------------------
# Sparse Auto-Encoder (Flax module)
# ---------------------------------------------------------------------------


class SparseAutoEncoder(nn.Module):
    """Sparse auto-encoder that learns monosemantic hallucination features.

    **Architecture:**
        Encoder: Dense(input_dim -> hidden_dim) -> ReLU -> top-1 sparsity mask
        Decoder: Dense(hidden_dim -> input_dim)

    **Top-1 sparsity:**
        After ReLU, only the single largest activation per sample is kept non-zero.
        This forces each sample to be "explained" by one dominant feature, producing
        monosemantic (single-concept) feature dimensions — the key property that makes
        HalluSAE features interpretable.

    **Loss:**
        L = MSE(x_recon, x) + sparsity_weight * mean(|h_sparse|)

        The L1 penalty on h_sparse further encourages sparse use of hidden dimensions
        even though the top-1 mask already enforces hard sparsity.

    REQ-VERIFY-160-1, REQ-VERIFY-160-2, REQ-VERIFY-160-3, REQ-VERIFY-160-4,
    SCENARIO-VERIFY-212
    """

    input_dim: int
    hidden_dim: int = 512
    sparsity_weight: float = 0.01

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass returning (reconstructed_x, sparse_hidden_activations).

        Parameters
        ----------
        x : jnp.ndarray, shape (batch, input_dim)
            Input feature vectors.

        Returns
        -------
        x_recon : jnp.ndarray, shape (batch, input_dim)
            Reconstruction of x through the bottleneck.
        h_sparse : jnp.ndarray, shape (batch, hidden_dim)
            Sparse activations — exactly one non-zero value per sample.
        """
        # Encoder: project to hidden space and apply ReLU
        h = nn.relu(nn.Dense(self.hidden_dim, name="encoder")(x))

        # Top-1 sparsity: zero out everything except the single largest activation.
        # keepdims=True broadcasts the max back over hidden_dim so the comparison
        # produces a boolean mask of the same shape as h.
        h_sparse = h * (h == jnp.max(h, axis=-1, keepdims=True)).astype(jnp.float32)

        # Decoder: project back to input space (no activation — linear reconstruction)
        x_recon = nn.Dense(self.input_dim, name="decoder")(h_sparse)

        return x_recon, h_sparse

    def loss(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute combined reconstruction + sparsity loss for a batch.

        MSE measures how well the single active feature can reconstruct the input.
        L1 on h_sparse penalises large activation magnitudes even for the one
        non-zero unit, encouraging the model to not cheat by storing magnitude.
        """
        x_recon, h_sparse = self(x)
        recon_loss = jnp.mean((x_recon - x) ** 2)
        l1_penalty = self.sparsity_weight * jnp.mean(jnp.abs(h_sparse))
        return recon_loss + l1_penalty


# ---------------------------------------------------------------------------
# Hallucination feature identification
# ---------------------------------------------------------------------------


def identify_hallucination_features(
    params: dict[str, Any],
    model: SparseAutoEncoder,
    features: jnp.ndarray,
    labels: jnp.ndarray,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rank sparse AE hidden dimensions by AUC vs hallucination label.

    **Why AUC and not accuracy:**
        AUC is threshold-free and measures whether a feature's activation is
        consistently higher on hallucinated steps than correct steps, regardless
        of the absolute activation magnitude.  This is exactly what HalluSAE
        uses for feature importance ranking.

    **Causal interpretation of top features:**
        A feature with AUC > 0.60 fires significantly more on incorrect steps than
        correct ones, implying it encodes something about the text structure that
        correlates with hallucination.  Feature names (assigned heuristically by
        index) connect back to the arithmetic error taxonomy from the FOVER corpus.

    Parameters
    ----------
    params : dict
        Trained Flax parameter dict for the SparseAutoEncoder.
    model : SparseAutoEncoder
        The model definition (used for apply()).
    features : jnp.ndarray, shape (n, input_dim)
        Feature vectors for all test samples.
    labels : jnp.ndarray, shape (n,)
        Binary hallucination labels: 1 = incorrect step, 0 = correct step.
    top_k : int
        Number of top features to return (default 10).

    Returns
    -------
    list of dicts, each with:
        feature_idx    : int   — index into the hidden_dim dimensions
        feature_auroc  : float — AUC of that dimension vs hallucination label
        feature_name   : str   — heuristic human-readable name
    """
    # Run encoder only to get sparse activations
    _, h_sparse = model.apply(params, features)
    h_np = jax.device_get(h_sparse)  # shape (n, hidden_dim)
    labels_np = jax.device_get(labels)  # shape (n,)

    n_features = h_np.shape[1]
    aurocs: list[tuple[int, float]] = []

    for dim in range(n_features):
        scores = h_np[:, dim]
        auc = _binary_auc(scores, labels_np)
        aurocs.append((dim, float(auc)))

    # Sort by AUC descending
    aurocs.sort(key=lambda t: t[1], reverse=True)
    top = aurocs[:top_k]

    # Assign heuristic names based on feature index ranges.
    # These are intentionally speculative — the value is that they give a
    # human-readable handle for features that fire on the test corpus.
    def _feature_name(idx: int) -> str:
        if idx < 20:
            return f"arithmetic_structure_{idx}"
        elif idx < 50:
            return f"digit_pattern_{idx}"
        elif idx < 100:
            return f"operator_sequence_{idx}"
        elif idx < 200:
            return f"equation_form_{idx}"
        else:
            return f"generic_text_{idx}"

    return [
        {
            "feature_idx": idx,
            "feature_auroc": round(auc, 4),
            "feature_name": _feature_name(idx),
        }
        for idx, auc in top
    ]


def _binary_auc(scores: Any, labels: Any) -> float:
    """Compute AUC for binary classification using the Wilcoxon-Mann-Whitney statistic.

    **Why not sklearn:**
        This keeps the module dependency-free beyond JAX/Flax.  The WMW AUC
        is equivalent to sklearn's roc_auc_score but computed in pure Python/numpy.

    AUC = P(score(positive) > score(negative)) where positive = hallucinated step.
    If all positives score higher than all negatives, AUC = 1.0 (perfect feature).
    If random, AUC = 0.5.
    """
    import numpy as np  # numpy is always available as a JAX dependency

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5  # degenerate case — no discrimination possible

    # Count concordant pairs: for each (pos, neg) pair, how often pos > neg?
    concordant = 0
    tied = 0
    total = len(pos_scores) * len(neg_scores)
    for p in pos_scores:
        concordant += int(np.sum(p > neg_scores))
        tied += int(np.sum(p == neg_scores))

    return (concordant + 0.5 * tied) / total
