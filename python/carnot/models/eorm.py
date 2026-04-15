"""EORM: Energy-based cOt Reward Model — JAX implementation.

**Researcher summary:**
    Implements an EORM-style CoT energy reward model following arXiv 2505.14999.
    A small transformer encoder reads the full (question, CoT response) pair and
    outputs a scalar energy: lower energy = model considers the CoT more correct.
    Training uses contrastive (hinge) loss on (correct, incorrect) response pairs.
    At inference, the best-of-N candidate is the one with the lowest energy.

**Detailed explanation for engineers:**
    Large Language Models generate chains of thought (CoT) — step-by-step
    reasoning traces — before producing a final answer.  These traces can be
    wrong: the model might hallucinate an intermediate calculation, apply the
    wrong formula, or lose track of units.

    The EORM ("Energy-based cOt Reward Model") addresses this by training a
    *verifier*: a second, smaller model that reads a complete CoT and predicts
    whether it is correct.  Instead of outputting a probability, the verifier
    outputs an energy — a number where *lower is better*.

    **Why energy instead of probability?**
    An energy function is unnormalized: we do not need to know the partition
    function (the normalizing constant that makes probabilities sum to 1).
    This makes training simpler.  We only need to know the *ordering*: correct
    CoTs should have lower energy than incorrect ones.  This is enforced by a
    contrastive (hinge) loss.

    **Architecture (following arXiv 2505.14999):**
    1. Concatenate question and response with a separator token.
    2. Tokenize: split on whitespace/punctuation, hash each word to a fixed
       vocabulary (no HuggingFace tokenizer needed — keeps CI fast on CPU).
    3. Embed: learnable token embeddings + learnable positional embeddings.
    4. Encode: N transformer layers (self-attention + feed-forward, pre-norm).
    5. Pool: mean over the sequence dimension → one vector per (question, CoT).
    6. Readout: linear projection → scalar energy.

    **Training:**
    Given a pair (correct_response, incorrect_response) for the same question,
    we want:
        E(correct) < E(incorrect)          [correct gets lower energy]

    The hinge loss penalizes violations:
        L = max(0, E(correct) - E(incorrect) + margin)

    L = 0 when E(incorrect) > E(correct) + margin (the margin provides a
    "safety gap" so the model does not merely learn to separate them by ε).
    Gradients flow through the transformer via JAX automatic differentiation.

    **Serialization:**
    All parameters are stored in a safetensors file (cross-language safe).
    Architecture config is stored in an adjacent JSON sidecar (`_config.json`).

    **CPU / CI safety:**
    The model defaults to embed_dim=128 so that unit tests run in seconds on
    CPU-only JAX.  No GPU, no HuggingFace tokenizer, no internet required.

Spec: REQ-LEARN-022, REQ-LEARN-023,
      SCENARIO-LEARN-038, SCENARIO-LEARN-039, SCENARIO-LEARN-040
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from safetensors.numpy import load_file, save_file


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Special token IDs (reserved at the bottom of the vocabulary)
_PAD_ID: int = 0   # padding — fills sequences shorter than max_seq_len
_SEP_ID: int = 1   # separator between question and response


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CoTEnergyInput:
    """Input to the EORM energy function: a (question, CoT response) pair.

    **For engineers:**
        The EORM model scores a *complete* chain-of-thought response in the
        context of the question that produced it.  We need both pieces of text
        because the same response text can be correct for one question and
        wrong for another.

        The two fields are concatenated internally with a [SEP] token:
            tokens = tokenize(question_text) + [SEP] + tokenize(response_text)

    Attributes:
        question_text: The original question asked to the LLM.
        response_text: The full CoT response (may include step-by-step reasoning
            and a final answer).

    Spec: REQ-LEARN-022-1
    """

    question_text: str
    response_text: str


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str, max_seq_len: int, vocab_size: int) -> list[int]:
    """Convert text to a list of token IDs using a hash-based word tokenizer.

    **For engineers:**
        This tokenizer avoids any external dependency (no HuggingFace, no
        sentencepiece).  It works in three steps:

        1. **Split**: Use a regex to extract all word-like tokens from the text,
           converting to lowercase.  Punctuation and whitespace are treated as
           separators and discarded.

        2. **Hash**: Map each word to an integer in [2, vocab_size) using a
           polynomial rolling hash (like Java's String.hashCode).  We reserve
           IDs 0 (padding) and 1 (separator) so all word tokens are ≥ 2.
           The hash is deterministic across runs (unlike Python's built-in
           ``hash()`` which randomizes per-process).

        3. **Truncate**: Return at most ``max_seq_len`` IDs.

    Args:
        text: Raw text to tokenize.
        max_seq_len: Maximum number of token IDs to return.
        vocab_size: Number of distinct token IDs (including special tokens).

    Returns:
        A list of integer token IDs in [0, vocab_size), length ≤ max_seq_len.
    """
    # Extract word-like tokens (alphanumeric runs), lowercase
    words = re.findall(r"[a-z0-9]+", text.lower())
    ids: list[int] = []
    for word in words:
        # Polynomial rolling hash — deterministic, no stdlib hash randomization
        h = 0
        for ch in word:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF  # keep in 32-bit unsigned
        # Map to [2, vocab_size) to avoid colliding with PAD and SEP
        ids.append(h % (vocab_size - 2) + 2)
    return ids[:max_seq_len]


def _make_token_sequence(
    question_text: str,
    response_text: str,
    max_seq_len: int,
    vocab_size: int,
) -> list[int]:
    """Build the full [question] + [SEP] + [response] token sequence.

    **For engineers:**
        The concatenation follows the standard BERT-style dual-encoder pattern:
        the separator token lets the self-attention layers learn different
        attention patterns for the question part versus the response part,
        even though both are encoded in a single forward pass.

        The sequence is truncated to ``max_seq_len`` *after* concatenation.
        If the question alone already fills the budget, the response may
        contribute zero tokens — this is an expected edge case for very long
        questions.

    Args:
        question_text: Question string.
        response_text: Response / CoT string.
        max_seq_len: Maximum total sequence length.
        vocab_size: Vocabulary size for hashing.

    Returns:
        Combined token ID list of length ≤ max_seq_len.
    """
    q_ids = _tokenize(question_text, max_seq_len, vocab_size)
    r_ids = _tokenize(response_text, max_seq_len, vocab_size)
    combined = q_ids + [_SEP_ID] + r_ids
    return combined[:max_seq_len]


# ---------------------------------------------------------------------------
# Parameter initialization helpers
# ---------------------------------------------------------------------------

def _init_layer(embed_dim: int, n_heads: int, key: jax.Array) -> dict[str, jax.Array]:
    """Initialize parameters for one transformer encoder layer.

    **For engineers:**
        A transformer layer has two sub-layers, each preceded by layer
        normalization (the "pre-norm" pattern):

        1. Multi-head self-attention:
           - Q, K, V linear projections (each embed_dim × embed_dim)
           - Output projection (embed_dim × embed_dim)
           - Layer-norm scale (gamma) and shift (beta)

        2. Position-wise feed-forward network (FFN):
           - Inner dimension = 4 × embed_dim (standard transformer expansion)
           - W1: embed_dim → 4*embed_dim, bias b1
           - W2: 4*embed_dim → embed_dim, bias b2
           - Layer-norm scale and shift

        Weights use Xavier uniform initialization (also called Glorot uniform):
            limit = sqrt(6 / (fan_in + fan_out))
            W ~ Uniform(-limit, +limit)

        This initialization keeps the variance of activations roughly constant
        through layers, preventing vanishing / exploding gradients at startup.

    Args:
        embed_dim: Model hidden dimension.
        n_heads: Number of attention heads.
        key: JAX PRNG key for weight initialization.

    Returns:
        Dictionary of named parameter arrays for one transformer layer.
    """
    d_ff = embed_dim * 4  # standard 4x expansion for the FFN inner dimension

    # Split the key into one sub-key per weight matrix
    k_q, k_k, k_v, k_o, k_f1, k_f2 = jrandom.split(key, 6)

    # Xavier initialization limits for each weight matrix
    attn_lim = jnp.sqrt(6.0 / (embed_dim + embed_dim))
    ff1_lim = jnp.sqrt(6.0 / (embed_dim + d_ff))
    ff2_lim = jnp.sqrt(6.0 / (d_ff + embed_dim))

    return {
        # --- Attention projection weights ---
        "w_q": jrandom.uniform(k_q, (embed_dim, embed_dim), minval=-attn_lim, maxval=attn_lim),
        "b_q": jnp.zeros(embed_dim),
        "w_k": jrandom.uniform(k_k, (embed_dim, embed_dim), minval=-attn_lim, maxval=attn_lim),
        "b_k": jnp.zeros(embed_dim),
        "w_v": jrandom.uniform(k_v, (embed_dim, embed_dim), minval=-attn_lim, maxval=attn_lim),
        "b_v": jnp.zeros(embed_dim),
        "w_o": jrandom.uniform(k_o, (embed_dim, embed_dim), minval=-attn_lim, maxval=attn_lim),
        "b_o": jnp.zeros(embed_dim),
        # --- Layer norm 1 (before attention) ---
        "ln1_gamma": jnp.ones(embed_dim),
        "ln1_beta": jnp.zeros(embed_dim),
        # --- Feed-forward network weights ---
        "w_ff1": jrandom.uniform(k_f1, (d_ff, embed_dim), minval=-ff1_lim, maxval=ff1_lim),
        "b_ff1": jnp.zeros(d_ff),
        "w_ff2": jrandom.uniform(k_f2, (embed_dim, d_ff), minval=-ff2_lim, maxval=ff2_lim),
        "b_ff2": jnp.zeros(embed_dim),
        # --- Layer norm 2 (before FFN) ---
        "ln2_gamma": jnp.ones(embed_dim),
        "ln2_beta": jnp.zeros(embed_dim),
    }


def _init_params(
    embed_dim: int,
    n_heads: int,
    n_layers: int,
    max_seq_len: int,
    vocab_size: int,
    key: jax.Array,
) -> dict[str, Any]:
    """Initialize all model parameters as a nested JAX pytree dict.

    **For engineers:**
        The returned dict is a valid JAX pytree, meaning ``jax.grad``,
        ``jax.tree_util.tree_map``, and ``jax.jit`` all work on it.
        Python dicts and lists are registered as pytree nodes by default.

        Structure::

            {
                "token_embed": (vocab_size, embed_dim),
                "pos_embed":   (max_seq_len, embed_dim),
                "layers": [ {layer_0_params}, {layer_1_params}, ... ],
                "final_ln_gamma": (embed_dim,),
                "final_ln_beta":  (embed_dim,),
                "out_weight":     (embed_dim,),
                "out_bias":       (1,),
            }

    Args:
        embed_dim: Model hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        max_seq_len: Maximum sequence length (positional embedding table size).
        vocab_size: Vocabulary size (token embedding table rows).
        key: Top-level JAX PRNG key.

    Returns:
        Nested dict of JAX arrays (a valid JAX pytree).
    """
    k_tok, k_pos, k_out, *layer_keys = jrandom.split(key, n_layers + 3)

    # Token embeddings — each token ID maps to an embed_dim-dimensional vector
    token_embed = jrandom.normal(k_tok, (vocab_size, embed_dim)) * 0.02

    # Positional embeddings — position i maps to an embed_dim vector
    # Small random init; the model learns to encode relative position during training
    pos_embed = jrandom.normal(k_pos, (max_seq_len, embed_dim)) * 0.02

    # Per-layer parameters (list of dicts — JAX handles lists as pytree nodes)
    layers = [_init_layer(embed_dim, n_heads, k) for k in layer_keys]

    # Final layer norm applied after the last transformer layer
    final_ln_gamma = jnp.ones(embed_dim)
    final_ln_beta = jnp.zeros(embed_dim)

    # Output head: dot product of pooled representation with a weight vector
    # Shape (embed_dim,) rather than (1, embed_dim) to produce a scalar directly
    out_limit = jnp.sqrt(6.0 / (embed_dim + 1))
    out_weight = jrandom.uniform(k_out, (embed_dim,), minval=-out_limit, maxval=out_limit)
    out_bias = jnp.zeros(1)

    return {
        "token_embed": token_embed,
        "pos_embed": pos_embed,
        "layers": layers,
        "final_ln_gamma": final_ln_gamma,
        "final_ln_beta": final_ln_beta,
        "out_weight": out_weight,
        "out_bias": out_bias,
    }


# ---------------------------------------------------------------------------
# Pure-functional forward pass (takes explicit params for jax.grad)
# ---------------------------------------------------------------------------

def _layer_norm(
    x: jax.Array,
    gamma: jax.Array,
    beta: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Layer normalization over the last axis.

    **For engineers:**
        Normalizes each token's feature vector to zero mean and unit variance,
        then applies a learnable affine transform (scale gamma, shift beta).

        Formula: output = gamma * (x - mean) / sqrt(var + eps) + beta

        Layer norm prevents activations from drifting during deep forward
        passes — crucial for stable transformer training.

    Args:
        x: Input array (..., d) — normalizes over the last dimension.
        gamma: Scale parameters, shape (d,).
        beta: Shift parameters, shape (d,).
        eps: Small constant for numerical stability (prevents division by zero).

    Returns:
        Normalized array, same shape as x.
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta


def _transformer_layer_forward(
    x: jax.Array,
    lp: dict[str, jax.Array],
    n_heads: int,
) -> jax.Array:
    """Forward pass through one transformer encoder layer.

    **For engineers:**
        Two sub-layers with residual connections (pre-norm variant):

        1. Pre-norm → multi-head self-attention → residual add
        2. Pre-norm → feed-forward network → residual add

        Multi-head self-attention splits embed_dim into n_heads groups
        (``d_head = embed_dim // n_heads`` per head).  Each head independently
        computes scaled dot-product attention:

            Attention(Q, K, V) = softmax(Q K^T / sqrt(d_head)) V

        The outputs of all heads are concatenated and projected back.

    Args:
        x: Token representations, shape (seq_len, embed_dim).
        lp: Layer parameter dict (keys: w_q, b_q, ..., ln2_gamma, ...).
        n_heads: Number of attention heads.

    Returns:
        Updated token representations, shape (seq_len, embed_dim).
    """
    seq_len, embed_dim = x.shape
    d_head = embed_dim // n_heads

    # ---------- Sub-layer 1: multi-head self-attention ----------
    x_norm = _layer_norm(x, lp["ln1_gamma"], lp["ln1_beta"])

    # Project to Q, K, V: (seq_len, embed_dim)
    q = x_norm @ lp["w_q"].T + lp["b_q"]
    k = x_norm @ lp["w_k"].T + lp["b_k"]
    v = x_norm @ lp["w_v"].T + lp["b_v"]

    # Reshape for multi-head: (seq_len, embed_dim) → (n_heads, seq_len, d_head)
    q = q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    k = k.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    v = v.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    # Scaled dot-product attention
    scale = jnp.sqrt(jnp.float32(d_head))
    scores = jnp.matmul(q, k.transpose(0, 2, 1)) / scale  # (n_heads, seq_len, seq_len)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.matmul(attn_weights, v)                 # (n_heads, seq_len, d_head)

    # Concatenate heads back: (n_heads, seq_len, d_head) → (seq_len, embed_dim)
    attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, embed_dim)

    # Output projection + residual
    x = x + attn_out @ lp["w_o"].T + lp["b_o"]

    # ---------- Sub-layer 2: feed-forward network ----------
    x_norm = _layer_norm(x, lp["ln2_gamma"], lp["ln2_beta"])
    ffn = jax.nn.gelu(x_norm @ lp["w_ff1"].T + lp["b_ff1"])
    ffn = ffn @ lp["w_ff2"].T + lp["b_ff2"]
    x = x + ffn

    return x


def _forward(
    params: dict[str, Any],
    token_ids: list[int],
    n_heads: int,
) -> jax.Array:
    """Pure-functional EORM forward pass.

    **For engineers:**
        This function is intentionally *pure* (no side effects, takes params
        explicitly) so that ``jax.grad(_forward, argnums=0)`` works.

        Pipeline:
        1. Look up token + positional embeddings, add them.
        2. Pass through each transformer layer sequentially.
        3. Apply final layer norm.
        4. Mean-pool over the sequence (one d_model vector per (q, r) pair).
        5. Dot with out_weight and add out_bias → scalar energy.

        **Why mean pooling vs CLS token?**
        Mean pooling uses information from every token position equally.
        A CLS token approach would require the first token to aggregate all
        context, which works but takes longer to learn.  Mean pooling is a
        simpler, equally effective default.

    Args:
        params: Nested dict of JAX arrays (pytree produced by _init_params).
        token_ids: List of integer token IDs (already truncated to max_seq_len).
        n_heads: Number of attention heads (needed to split embed_dim per head).

    Returns:
        Scalar JAX array representing the energy for this (question, CoT) pair.
    """
    seq_len = len(token_ids)

    # --- 1. Embed tokens + positions ---
    # token_ids[i] indexes into the embedding table to get a d_model vector
    token_ids_arr = jnp.array(token_ids, dtype=jnp.int32)
    pos_ids = jnp.arange(seq_len, dtype=jnp.int32)

    x = params["token_embed"][token_ids_arr] + params["pos_embed"][pos_ids]
    # x shape: (seq_len, embed_dim)

    # --- 2. Transformer layers ---
    for lp in params["layers"]:
        x = _transformer_layer_forward(x, lp, n_heads)

    # --- 3. Final layer norm ---
    x = _layer_norm(x, params["final_ln_gamma"], params["final_ln_beta"])

    # --- 4. Mean pool over sequence ---
    pooled = jnp.mean(x, axis=0)  # (embed_dim,)

    # --- 5. Linear readout → scalar energy ---
    energy = jnp.dot(pooled, params["out_weight"]) + params["out_bias"][0]

    return energy


# ---------------------------------------------------------------------------
# Parameter count helpers
# ---------------------------------------------------------------------------

def _count_params(params: dict[str, Any]) -> int:
    """Count total number of scalar parameters in a nested pytree dict.

    **For engineers:**
        ``jax.tree_util.tree_leaves`` flattens any pytree (nested dicts, lists,
        tuples) into a list of leaf arrays.  We sum the size of each leaf array
        to get the total parameter count.  This is equivalent to what PyTorch's
        ``model.numel()`` does.

    Args:
        params: Nested dict of JAX arrays.

    Returns:
        Total number of scalar float parameters.
    """
    return sum(arr.size for arr in jax.tree_util.tree_leaves(params))


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, jax.Array]:
    """Flatten a nested param dict to a flat string-keyed dict for safetensors.

    **For engineers:**
        safetensors requires a flat ``{str: np.ndarray}`` mapping.  We convert
        the nested structure (dicts and lists) to flat keys using "/" as a
        separator, e.g.:
            ``params["layers"][0]["w_q"]``  →  key ``"layers/0/w_q"``

        This is fully reversible via ``_unflatten_params``.

    Args:
        params: Nested dict / list of JAX arrays.
        prefix: Key prefix accumulated by recursive calls.

    Returns:
        Flat dict mapping string keys to JAX arrays.
    """
    flat: dict[str, jax.Array] = {}
    if isinstance(params, dict):
        for k, v in params.items():
            child_prefix = f"{prefix}/{k}" if prefix else k
            flat.update(_flatten_params(v, child_prefix))
    elif isinstance(params, list):
        for i, v in enumerate(params):
            child_prefix = f"{prefix}/{i}" if prefix else str(i)
            flat.update(_flatten_params(v, child_prefix))
    else:
        # Leaf — a JAX array
        flat[prefix] = params
    return flat


def _unflatten_params(
    flat: dict[str, Any],
    template: dict[str, Any],
) -> dict[str, Any]:
    """Restore a nested param dict from a flat dict using a template for structure.

    **For engineers:**
        We use the template (produced by ``_init_params``) to know the expected
        nested structure.  Then we re-fill the leaves with values from the flat
        dict.  This avoids storing the structure itself (it is deterministic
        given the config).

    Args:
        flat: Flat dict from ``_flatten_params`` with values converted to JAX.
        template: Nested dict of JAX arrays with the desired structure.

    Returns:
        Nested dict matching the structure of ``template`` with values from ``flat``.
    """
    def _restore(node: Any, prefix: str) -> Any:
        if isinstance(node, dict):
            return {
                k: _restore(v, f"{prefix}/{k}" if prefix else k)
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [
                _restore(v, f"{prefix}/{i}" if prefix else str(i))
                for i, v in enumerate(node)
            ]
        # Leaf: look up in flat dict
        return flat[prefix]

    return _restore(template, "")


# ---------------------------------------------------------------------------
# EORMModel
# ---------------------------------------------------------------------------

class EORMModel:
    """EORM CoT energy reward model — scores chain-of-thought responses by energy.

    **Researcher summary:**
        Transformer encoder (n_layers=2, embed_dim=128, 4 heads) that maps a
        (question, CoT response) pair to a scalar energy.  Lower energy = model
        considers the CoT more likely to be correct.  Intended for best-of-N
        selection: score N candidates and pick the one with the lowest energy.

    **Detailed explanation for engineers:**
        This model follows arXiv 2505.14999 ("EORM").  The key design choices:

        - **No HuggingFace tokenizer**: Uses a deterministic hash-based word
          tokenizer so CI tests run on CPU in seconds without downloading anything.

        - **Pure JAX (no Flax)**: All parameters are stored in a plain Python
          dict (a valid JAX pytree).  This makes it easy to take gradients via
          ``jax.grad``, serialise via safetensors, and read the code without
          knowing any framework-specific APIs.

        - **CPU-safe**: The default config (embed_dim=128, n_layers=2) is small
          enough to run comfortably on a laptop CPU.

    For example::

        model = EORMModel()
        cot = CoTEnergyInput(question_text="What is 2+2?", response_text="It is 4.")
        energy = model.energy(cot)            # scalar float

        best_idx = model.rank(["wrong", "right"], question="2+2?")[0]

    Spec: REQ-LEARN-022
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 512,
        vocab_size: int = 4096,
        key: jax.Array | None = None,
    ) -> None:
        """Create an EORMModel with randomly initialized parameters.

        **For engineers:**
            All architecture hyperparameters are stored so they can be written
            to the config sidecar during ``save()`` and restored during
            ``load()``.

            ``embed_dim`` must be divisible by ``n_heads`` so that each
            attention head gets an equal ``d_head = embed_dim // n_heads``
            slice of the embedding.

        Args:
            embed_dim: Hidden dimension for token embeddings and all internal
                representations.  Default 128.
            n_heads: Number of self-attention heads.  Must divide embed_dim.
                Default 4.
            n_layers: Number of transformer encoder layers.  Default 2.
            max_seq_len: Maximum token sequence length.  Inputs longer than
                this are silently truncated.  Default 512.
            vocab_size: Number of distinct token IDs in the vocabulary
                (including special tokens PAD=0, SEP=1).  Default 4096.
            key: JAX PRNG key for weight initialization.  Uses PRNGKey(0)
                if None.

        Raises:
            ValueError: If embed_dim is not divisible by n_heads.

        Spec: REQ-LEARN-022-2
        """
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
            )

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        if key is None:
            key = jrandom.PRNGKey(0)

        self.params = _init_params(embed_dim, n_heads, n_layers, max_seq_len, vocab_size, key)

    # ------------------------------------------------------------------
    # Core energy computation
    # ------------------------------------------------------------------

    def energy(self, cot_input: CoTEnergyInput) -> float:
        """Compute the energy of a (question, CoT response) pair.

        **For engineers:**
            Lower energy = model considers this CoT more likely to be correct.
            The energy is *not* a probability — it is an unnormalized scalar.
            To compare two responses for the same question, compute both
            energies and pick the lower one.

            Pipeline:
            1. Build token sequence: tokenize(question) + [SEP] + tokenize(response)
            2. Run through transformer encoder
            3. Mean-pool → scalar via linear readout

        Args:
            cot_input: Dataclass holding question_text and response_text.

        Returns:
            A Python float representing the energy.  Lower is better.

        Spec: REQ-LEARN-022-3, SCENARIO-LEARN-038
        """
        token_ids = _make_token_sequence(
            cot_input.question_text,
            cot_input.response_text,
            self.max_seq_len,
            self.vocab_size,
        )
        # Fall back to a single SEP token if both texts tokenize to nothing
        if not token_ids:
            token_ids = [_SEP_ID]
        return float(_forward(self.params, token_ids, self.n_heads))

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(self, responses: list[str], question: str) -> list[int]:
        """Rank response candidates by energy — lowest energy first.

        **For engineers:**
            Scores every response independently (using the same question text)
            and returns the indices sorted so the best (lowest energy) response
            is at position 0.  This is the core best-of-N selection operation.

            Example::

                best = responses[model.rank(responses, question)[0]]

        Args:
            responses: List of candidate response strings to score and rank.
            question: The question that produced these responses.

        Returns:
            List of integer indices into ``responses``, sorted ascending by
            energy (index of the best response is first).

        Spec: REQ-LEARN-022-4, SCENARIO-LEARN-039
        """
        energies = [
            self.energy(CoTEnergyInput(question_text=question, response_text=r))
            for r in responses
        ]
        # argsort: returns indices that would sort energies in ascending order
        return sorted(range(len(energies)), key=lambda i: energies[i])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model parameters to a safetensors file with a JSON config sidecar.

        **For engineers:**
            Writes two files:
            - ``<path>``: safetensors binary with all parameter tensors.
            - ``<path.stem>_config.json`` in the same directory: JSON with the
              architecture config (embed_dim, n_heads, n_layers, max_seq_len,
              vocab_size) so the model can be reconstructed on load.

            The safetensors format is:
            - Safe: cannot execute arbitrary code on load (unlike pickle).
            - Cross-language: readable by the Rust safetensors crate.
            - Efficient: memory-mappable, supports partial loading.

        Args:
            path: Full file path for the .safetensors output file.

        Spec: REQ-LEARN-022-5
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten nested param pytree → flat {str: np.ndarray} for safetensors
        flat = _flatten_params(self.params)
        np_flat = {k: np.asarray(v) for k, v in flat.items()}
        save_file(np_flat, str(path))

        # Write architecture config as a JSON sidecar so load() can reconstruct
        config = {
            "embed_dim": self.embed_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
        }
        config_path = path.parent / f"{path.stem}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "EORMModel":
        """Load a saved EORMModel from a safetensors file and config sidecar.

        **For engineers:**
            Reads the ``_config.json`` sidecar to reconstruct the architecture,
            then creates a new model and overwrites its parameters from the
            safetensors file.

            Because ``_unflatten_params`` needs the nested template structure,
            we first create a fresh model (with random weights) and then
            replace its params.

        Args:
            path: Full file path to the ``.safetensors`` file written by
                ``save()``.

        Returns:
            A fully reconstructed EORMModel with the saved parameters.

        Spec: REQ-LEARN-022-5
        """
        path = Path(path)
        config_path = path.parent / f"{path.stem}_config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Build a fresh model with the correct architecture (provides template)
        model = cls(**config)

        # Load flat params from safetensors and convert NumPy → JAX arrays
        np_flat = load_file(str(path))
        jax_flat = {k: jnp.array(v) for k, v in np_flat.items()}

        # Restore nested pytree structure using the fresh model's params as template
        model.params = _unflatten_params(jax_flat, model.params)
        return model

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Total number of trainable scalar parameters.

        **For engineers:**
            Useful for sanity-checking that the model is the right size before
            training.  The EORM paper uses a 55M-parameter model; our default
            (embed_dim=128, n_layers=2) is much smaller and CPU-friendly.

        Returns:
            Integer count of all scalar float parameters.

        Spec: REQ-LEARN-022-6, SCENARIO-LEARN-038
        """
        return _count_params(self.params)


# ---------------------------------------------------------------------------
# EORMTrainer
# ---------------------------------------------------------------------------

class EORMTrainer:
    """Trains an EORMModel via contrastive (hinge) loss on CoT response pairs.

    **Researcher summary:**
        Contrastive training: for each (correct_response, incorrect_response,
        question) triple, push E(correct) down and E(incorrect) up.
        Loss = max(0, E(correct) - E(incorrect) + margin).
        Gradients computed via jax.value_and_grad, SGD update.

    **Detailed explanation for engineers:**
        Contrastive loss is the right choice for a *ranking* problem.  We do not
        know the absolute "correct" energy for a response — we only know which
        response is better.  The hinge loss enforces the relative ordering:

            E(correct) + margin < E(incorrect)

        If E(incorrect) is already larger than E(correct) + margin, the loss is
        zero and the gradient is zero — the model has already learned to
        distinguish this pair and should not be perturbed further.

        Training uses plain gradient descent (no momentum, no adaptive LR) for
        simplicity.  The model's params dict is mutated in place after each step.

    For example::

        model = EORMModel(embed_dim=64, n_layers=1)   # small for fast testing
        trainer = EORMTrainer(model)
        pairs = [("The answer is 4.", "The answer is 5.", "What is 2+2?")]
        loss = trainer.train_epoch(pairs)

    Spec: REQ-LEARN-023
    """

    def __init__(self, model: EORMModel, lr: float = 1e-4, margin: float = 1.0) -> None:
        """Create an EORMTrainer.

        Args:
            model: The EORMModel to train.  Parameters are modified in place.
            lr: Learning rate for gradient descent.  Default 1e-4.
            margin: Hinge loss margin.  The model is penalized when
                E(incorrect) - E(correct) < margin.  Default 1.0.

        Spec: REQ-LEARN-023-1
        """
        self.model = model
        self.lr = lr
        self.margin = margin

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def contrastive_loss(
        self,
        correct_energy: float,
        incorrect_energy: float,
    ) -> float:
        """Compute hinge contrastive loss: max(0, E_correct - E_incorrect + margin).

        **For engineers:**
            Hinge loss is zero when the margin condition is already satisfied
            (E_incorrect > E_correct + margin), and positive otherwise.

            When positive, the gradient pushes:
            - E(correct) down (make the correct response look better)
            - E(incorrect) up (make the incorrect response look worse)

            This is a pure Python function for readability; the actual gradient
            computation in ``train_step`` uses JAX so it operates on JAX arrays.

        Args:
            correct_energy: Energy of the correct response.
            incorrect_energy: Energy of the incorrect response.

        Returns:
            Scalar float loss value ≥ 0.

        Spec: REQ-LEARN-023-2, SCENARIO-LEARN-040
        """
        return float(max(0.0, correct_energy - incorrect_energy + self.margin))

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        correct_response: str,
        incorrect_response: str,
        question: str,
    ) -> float:
        """Run one gradient update on a (correct, incorrect, question) triple.

        **For engineers:**
            Uses ``jax.value_and_grad`` to compute the loss *and* its gradient
            w.r.t. the model parameters in a single forward+backward pass:

                loss, grads = jax.value_and_grad(loss_fn)(params)

            Then applies a simple SGD update:

                params = params - lr * grads

            ``jax.tree_util.tree_map`` applies this elementwise to every array
            in the nested params pytree simultaneously.

        Args:
            correct_response: A CoT response that is correct for this question.
            incorrect_response: A CoT response that is incorrect.
            question: The question that produced both responses.

        Returns:
            The scalar loss value before the update (Python float).

        Spec: REQ-LEARN-023-3
        """
        # Build token sequences outside the differentiable function
        n_heads = self.model.n_heads
        correct_ids = _make_token_sequence(
            question, correct_response, self.model.max_seq_len, self.model.vocab_size
        ) or [_SEP_ID]
        incorrect_ids = _make_token_sequence(
            question, incorrect_response, self.model.max_seq_len, self.model.vocab_size
        ) or [_SEP_ID]

        def loss_fn(params: dict[str, Any]) -> jax.Array:
            """Contrastive hinge loss as a pure function of params."""
            e_correct = _forward(params, correct_ids, n_heads)
            e_incorrect = _forward(params, incorrect_ids, n_heads)
            # jax.nn.relu is the differentiable equivalent of max(0, x)
            return jax.nn.relu(e_correct - e_incorrect + self.margin)

        loss_val, grads = jax.value_and_grad(loss_fn)(self.model.params)

        # SGD update: move params in the direction that reduces loss
        self.model.params = jax.tree_util.tree_map(
            lambda p, g: p - self.lr * g,
            self.model.params,
            grads,
        )

        return float(loss_val)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        pairs: list[tuple[str, str, str]],
        batch_size: int = 16,
    ) -> float:
        """Train for one epoch over a list of (correct, incorrect, question) pairs.

        **For engineers:**
            Iterates over ``pairs`` in chunks of ``batch_size`` (this allows
            future batching optimizations; currently each pair is a single
            ``train_step``).  Returns the mean loss across all pairs so callers
            can monitor convergence.

            If ``pairs`` is empty, returns 0.0.

        Args:
            pairs: List of (correct_response, incorrect_response, question) triples.
            batch_size: Chunk size for iteration (used for future batching; each
                pair is still processed individually in the current implementation).

        Returns:
            Mean training loss over all pairs in this epoch.

        Spec: REQ-LEARN-023-4
        """
        if not pairs:
            return 0.0

        total_loss = 0.0
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            for correct_resp, incorrect_resp, question in chunk:
                total_loss += self.train_step(correct_resp, incorrect_resp, question)

        return total_loss / len(pairs)
