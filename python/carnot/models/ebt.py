"""Energy-Based Transformer (EBT) — JAX implementation.

**Researcher summary:**
    Implements the Energy-Based Transformer from arxiv 2507.02092. The model
    is a standard transformer that outputs a scalar energy instead of
    next-token logits. Inference is performed via gradient descent on the
    energy surface: given fixed input tokens, we optimize candidate output
    tokens to minimize E(input, output).

    Architecture: token_embed + pos_embed → N × [MultiHeadSelfAttention + FFN]
    → mean_pool → linear → scalar energy.

**Detailed explanation for engineers:**
    A conventional transformer (like GPT) outputs a probability distribution
    over the next token. An Energy-Based Transformer instead outputs a single
    scalar number — the "energy" — for an entire (input, output) sequence.
    Low energy means the model considers that (input, output) pair likely;
    high energy means unlikely.

    **How inference works (the key difference from GPT):**
    Instead of generating tokens left-to-right via sampling, EBT inference
    works by optimization:
    1. Concatenate input_tokens and candidate_output_tokens into one sequence.
    2. Compute E(sequence) — forward pass through the transformer.
    3. Compute dE/d(output_token_embeddings) via backprop.
    4. Update the candidate outputs to lower the energy.
    5. Repeat until convergence.

    This allows the model to "think globally" about the entire output at once,
    rather than committing to tokens one at a time.

    **Architecture details:**
    - Token embeddings: learnable lookup table, vocab_size × d_model
    - Positional embeddings: learnable, max_seq_len × d_model
    - Each transformer layer has:
      - Multi-head self-attention (Q, K, V projections → scaled dot-product
        attention → concatenate heads → output projection)
      - Feed-forward network (linear → GELU → linear)
      - Layer normalization before each sub-layer (pre-norm)
    - After all layers: mean-pool over sequence → linear → scalar energy

    **Why pure JAX (no Flax/Equinox)?**
    Keeps the implementation self-contained and transparent. Every matrix
    multiply and attention computation is explicit — no hidden abstractions.
    This makes it easier to understand, debug, and port to Rust later.

Spec: REQ-EBT-001, REQ-EBT-002, REQ-EBT-003
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import AutoGradMixin


def _gelu(x: jax.Array) -> jax.Array:
    """GELU activation: Gaussian Error Linear Unit.

    **For engineers:**
        GELU is the standard activation in modern transformers (BERT, GPT).
        It is a smooth approximation of ReLU that allows small negative values
        through, which helps with gradient flow.

        Formula: x * Phi(x) where Phi is the standard Gaussian CDF.
        JAX provides an optimized implementation.
    """
    return jax.nn.gelu(x)


def _layer_norm(x: jax.Array, gamma: jax.Array, beta: jax.Array) -> jax.Array:
    """Layer normalization: normalize across the feature dimension.

    **For engineers:**
        Layer norm stabilizes transformer training by normalizing each
        token's feature vector to zero mean and unit variance, then
        applying a learnable scale (gamma) and shift (beta).

        For a vector x of shape (d_model,):
            x_norm = (x - mean(x)) / sqrt(var(x) + eps)
            output = gamma * x_norm + beta

        This prevents activations from growing unboundedly through
        deep transformer layers.

    Args:
        x: Input array of shape (seq_len, d_model) or (d_model,).
        gamma: Scale parameter of shape (d_model,).
        beta: Shift parameter of shape (d_model,).

    Returns:
        Normalized array, same shape as x.
    """
    eps = 1e-5
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return gamma * x_norm + beta


@dataclass
class EBTConfig:
    """Configuration for the Energy-Based Transformer.

    **Researcher summary:**
        Specifies transformer architecture: depth, width, attention heads,
        feed-forward expansion, vocabulary size, and maximum sequence length.

    **Detailed explanation for engineers:**
        These are the architectural hyperparameters you choose before
        creating an EBT model. They cannot be changed after initialization.

        - ``n_layers``: Number of transformer layers (depth). More layers =
          more expressive but slower and more memory. Typical: 2-12.

        - ``d_model``: Hidden dimension size (width). Every token is
          represented as a vector of this size throughout the network.
          Must be divisible by n_heads. Typical: 64-512.

        - ``n_heads``: Number of attention heads. Multi-head attention
          lets the model attend to different positions for different
          "reasons" simultaneously. d_model must be divisible by n_heads.

        - ``d_ff``: Feed-forward network inner dimension. Typically 4x
          d_model. The FFN in each layer is: linear(d_model → d_ff) →
          GELU → linear(d_ff → d_model).

        - ``vocab_size``: Number of distinct tokens the model can handle.
          Each token ID maps to a learnable embedding vector.

        - ``max_seq_len``: Maximum sequence length (input + output tokens
          combined). Positional embeddings are allocated for this many
          positions.

    Spec: REQ-EBT-001
    """

    n_layers: int = 4
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    vocab_size: int = 256
    max_seq_len: int = 128

    def validate(self) -> None:
        """Validate configuration parameters.

        **For engineers:**
            Checks that all dimensions are positive and that d_model is
            evenly divisible by n_heads (required for multi-head attention
            to split the embedding dimension across heads).

        Raises:
            ValueError: If any parameter is invalid.

        Spec: SCENARIO-EBT-001
        """
        if self.n_layers <= 0:
            raise ValueError("n_layers must be > 0")
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be > 0")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )


class _TransformerLayer:
    """A single transformer layer: pre-norm self-attention + pre-norm FFN.

    **For engineers:**
        Each transformer layer has two sub-layers:

        1. **Multi-head self-attention** (with pre-norm):
           - Layer-normalize the input
           - Project to Q, K, V matrices (one set per attention head)
           - Compute scaled dot-product attention per head
           - Concatenate heads and project back to d_model
           - Add residual connection (output = attention_out + input)

        2. **Feed-forward network** (with pre-norm):
           - Layer-normalize the input
           - Linear d_model → d_ff, then GELU, then linear d_ff → d_model
           - Add residual connection (output = ffn_out + input)

        **Pre-norm vs post-norm:**
        We normalize BEFORE each sub-layer (pre-norm), which is more stable
        for training deep transformers than the original post-norm approach.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, key: jax.Array) -> None:
        """Initialize transformer layer parameters.

        Args:
            d_model: Model hidden dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward inner dimension.
            key: JAX PRNG key for weight initialization.
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        k_q, k_k, k_v, k_o, k_f1, k_f2, key = jrandom.split(key, 7)

        # Xavier initialization scale for attention projections
        attn_limit = jnp.sqrt(6.0 / (d_model + d_model))

        # Attention projections: Q, K, V, and output
        # Shape: (d_model, d_model) — projects full d_model to full d_model,
        # then we reshape to (n_heads, d_head) for multi-head computation
        self.w_q = jrandom.uniform(k_q, (d_model, d_model), minval=-attn_limit, maxval=attn_limit)
        self.w_k = jrandom.uniform(k_k, (d_model, d_model), minval=-attn_limit, maxval=attn_limit)
        self.w_v = jrandom.uniform(k_v, (d_model, d_model), minval=-attn_limit, maxval=attn_limit)
        self.w_o = jrandom.uniform(k_o, (d_model, d_model), minval=-attn_limit, maxval=attn_limit)

        # Attention biases
        self.b_q = jnp.zeros(d_model)
        self.b_k = jnp.zeros(d_model)
        self.b_v = jnp.zeros(d_model)
        self.b_o = jnp.zeros(d_model)

        # Layer norm parameters for attention sub-layer
        self.ln1_gamma = jnp.ones(d_model)
        self.ln1_beta = jnp.zeros(d_model)

        # Feed-forward network weights
        ff_limit1 = jnp.sqrt(6.0 / (d_model + d_ff))
        ff_limit2 = jnp.sqrt(6.0 / (d_ff + d_model))
        self.w_ff1 = jrandom.uniform(k_f1, (d_ff, d_model), minval=-ff_limit1, maxval=ff_limit1)
        self.b_ff1 = jnp.zeros(d_ff)
        self.w_ff2 = jrandom.uniform(k_f2, (d_model, d_ff), minval=-ff_limit2, maxval=ff_limit2)
        self.b_ff2 = jnp.zeros(d_model)

        # Layer norm parameters for FFN sub-layer
        self.ln2_gamma = jnp.ones(d_model)
        self.ln2_beta = jnp.zeros(d_model)

    def forward(self, x: jax.Array) -> jax.Array:
        """Forward pass through the transformer layer.

        **For engineers:**
            The computation flow is:

            1. Pre-norm → Multi-head attention → Residual add
            2. Pre-norm → FFN → Residual add

            Multi-head attention splits the d_model dimension into n_heads
            groups of d_head dimensions each. Each head independently
            computes attention, allowing the model to focus on different
            aspects of the input simultaneously.

            Scaled dot-product attention:
                attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_head)) @ V

            The scaling by sqrt(d_head) prevents the dot products from
            growing too large, which would push softmax into regions with
            tiny gradients.

        Args:
            x: Input tensor of shape (seq_len, d_model).

        Returns:
            Output tensor of shape (seq_len, d_model).
        """
        seq_len = x.shape[0]

        # --- Sub-layer 1: Multi-head self-attention with pre-norm ---
        x_norm = _layer_norm(x, self.ln1_gamma, self.ln1_beta)

        # Project to Q, K, V: (seq_len, d_model) @ (d_model, d_model)^T → (seq_len, d_model)
        q = x_norm @ self.w_q.T + self.b_q  # (seq_len, d_model)
        k = x_norm @ self.w_k.T + self.b_k
        v = x_norm @ self.w_v.T + self.b_v

        # Reshape for multi-head: (seq_len, d_model) → (seq_len, n_heads, d_head)
        #                         → (n_heads, seq_len, d_head)
        q = q.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        k = k.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        v = v.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)

        # Scaled dot-product attention per head:
        # scores: (n_heads, seq_len, seq_len)
        scale = jnp.sqrt(jnp.float32(self.d_head))
        scores = jnp.matmul(q, k.transpose(0, 2, 1)) / scale
        attn_weights = jax.nn.softmax(scores, axis=-1)

        # Apply attention to values: (n_heads, seq_len, d_head)
        attn_out = jnp.matmul(attn_weights, v)

        # Concatenate heads: (n_heads, seq_len, d_head) → (seq_len, d_model)
        attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, self.d_model)

        # Output projection + residual connection
        attn_out = attn_out @ self.w_o.T + self.b_o
        x = x + attn_out

        # --- Sub-layer 2: Feed-forward network with pre-norm ---
        x_norm = _layer_norm(x, self.ln2_gamma, self.ln2_beta)

        # FFN: linear → GELU → linear
        ffn_out = _gelu(x_norm @ self.w_ff1.T + self.b_ff1)
        ffn_out = ffn_out @ self.w_ff2.T + self.b_ff2

        # Residual connection
        x = x + ffn_out

        return x


class EBTransformer(AutoGradMixin):
    """Energy-Based Transformer — outputs scalar energy for token sequences.

    **Researcher summary:**
        Transformer architecture from arxiv 2507.02092 that maps a sequence
        of token IDs to a scalar energy. E(x) = linear(mean_pool(transformer(embed(x)))).
        Inference via gradient descent on the energy surface.

    **Detailed explanation for engineers:**
        This model takes a 1-D integer array of token IDs (typically the
        concatenation of input tokens and candidate output tokens) and
        produces a single scalar energy value.

        **Forward pass:**
        1. Look up token embeddings from the embedding table
        2. Add positional embeddings (so the model knows token order)
        3. Pass through N transformer layers (self-attention + FFN each)
        4. Apply final layer normalization
        5. Mean-pool across the sequence dimension → single d_model vector
        6. Linear projection → scalar energy

        **Why mean pooling?**
        We need to reduce a variable-length sequence of vectors to a single
        vector. Mean pooling (averaging all token representations) is simple,
        differentiable, and works well in practice. Unlike [CLS] token
        approaches, it uses information from all positions equally.

        **The energy interface:**
        Because this inherits from AutoGradMixin, the energy() method takes
        a 1-D float array. For token inputs, we cast integer token IDs to
        float for the energy computation (the embeddings are looked up via
        integer indexing internally). The grad_energy() and energy_batch()
        methods are automatically provided by JAX autodiff and vmap.

    For example::

        config = EBTConfig(n_layers=2, d_model=32, n_heads=4, d_ff=64,
                           vocab_size=100, max_seq_len=16)
        model = EBTransformer(config)
        tokens = jnp.array([1, 5, 23, 7, 42])  # 5 token IDs
        e = model.energy(tokens.astype(jnp.float32))  # scalar energy

    Spec: REQ-EBT-002, REQ-EBT-003
    """

    def __init__(
        self,
        config: EBTConfig,
        key: jax.Array | None = None,
    ) -> None:
        """Create a new Energy-Based Transformer with initialized parameters.

        **Detailed explanation for engineers:**
            Validates config, then builds:
            1. Token embedding table: (vocab_size, d_model) — each row is the
               learnable vector for one token ID.
            2. Positional embedding table: (max_seq_len, d_model) — each row
               encodes position information.
            3. N transformer layers, each with attention + FFN parameters.
            4. Final layer norm parameters.
            5. Output projection: d_model → 1 (scalar energy readout).

            All weights use Xavier/Glorot uniform initialization.
            Embeddings use normal initialization with std=0.02 (standard
            for transformers, following GPT-2).

        Args:
            config: EBTConfig specifying architecture.
            key: JAX PRNG key for random initialization. If None, uses seed 0.

        Raises:
            ValueError: If config has invalid values.

        Spec: REQ-EBT-002, SCENARIO-EBT-001
        """
        config.validate()
        self.config = config

        if key is None:
            key = jrandom.PRNGKey(0)

        # Token embeddings: (vocab_size, d_model)
        # Normal initialization with std=0.02 is standard for transformer embeddings
        k_tok, k_pos, key = jrandom.split(key, 3)
        self.token_embeddings = jrandom.normal(k_tok, (config.vocab_size, config.d_model)) * 0.02

        # Positional embeddings: (max_seq_len, d_model)
        self.pos_embeddings = jrandom.normal(k_pos, (config.max_seq_len, config.d_model)) * 0.02

        # Transformer layers
        self.layers: list[_TransformerLayer] = []
        for _ in range(config.n_layers):
            k_layer, key = jrandom.split(key)
            self.layers.append(
                _TransformerLayer(config.d_model, config.n_heads, config.d_ff, k_layer)
            )

        # Final layer norm (applied after all transformer layers)
        self.final_ln_gamma = jnp.ones(config.d_model)
        self.final_ln_beta = jnp.zeros(config.d_model)

        # Output projection: d_model → scalar energy
        # Xavier-initialized so different inputs produce different energies
        # at initialization (unlike Gibbs/Boltzmann which zero-init this).
        # For a transformer, zero output weights would waste all the
        # expressive power of the attention layers.
        k_out, key = jrandom.split(key)
        out_limit = jnp.sqrt(6.0 / (config.d_model + 1))
        self.output_weight = jrandom.uniform(
            k_out, (config.d_model,), minval=-out_limit, maxval=out_limit
        )
        self.output_bias = 0.0

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute scalar energy E(x) for a token sequence.

        **Researcher summary:**
            Forward pass: embed(x) → transformer layers → mean_pool → linear → scalar.

        **Detailed explanation for engineers:**
            The input x is a 1-D array of token IDs (as floats, since the
            AutoGradMixin interface uses float arrays). The token IDs are
            rounded to integers for embedding lookup.

            Step by step:
            1. Round float values to nearest integer for embedding lookup
            2. Clip to valid vocab range [0, vocab_size - 1]
            3. Look up token embeddings → (seq_len, d_model)
            4. Add positional embeddings for positions 0..seq_len-1
            5. Pass through each transformer layer sequentially
            6. Apply final layer normalization
            7. Mean-pool across sequence: (seq_len, d_model) → (d_model,)
            8. Linear projection to scalar: w @ h + b → E(x)

            **Note on differentiability:**
            The embedding lookup (integer indexing) is not differentiable
            with respect to the token IDs themselves. This is by design —
            in EBT inference, we optimize in the continuous embedding space,
            not over discrete tokens. The energy() method here is used for
            evaluation; for inference optimization, you would work with
            continuous embeddings directly.

        Args:
            x: A 1-D JAX array of token IDs (as float32). Shape: (seq_len,).

        Returns:
            A scalar JAX array representing the energy.

        Spec: REQ-EBT-002, SCENARIO-EBT-002
        """
        # Convert float token IDs to integer indices for embedding lookup
        token_ids = jnp.clip(jnp.round(x).astype(jnp.int32), 0, self.config.vocab_size - 1)
        seq_len = token_ids.shape[0]

        # Token embeddings + positional embeddings
        h = self.token_embeddings[token_ids] + self.pos_embeddings[:seq_len]

        # Pass through transformer layers
        for layer in self.layers:
            h = layer.forward(h)

        # Final layer normalization
        h = _layer_norm(h, self.final_ln_gamma, self.final_ln_beta)

        # Mean pool across sequence dimension: (seq_len, d_model) → (d_model,)
        h_pooled = jnp.mean(h, axis=0)

        # Linear projection to scalar energy
        return self.output_weight @ h_pooled + self.output_bias

    @property
    def input_dim(self) -> int:
        """Maximum sequence length (number of token positions).

        **For engineers:**
            For the EBT, input_dim represents max_seq_len since the input
            is a variable-length sequence of tokens. The actual input can
            be shorter than this.
        """
        return self.config.max_seq_len

    def parameter_memory_bytes(self) -> int:
        """Compute total memory footprint of model parameters in bytes.

        **For engineers:**
            Sums up memory for all parameter tensors: embeddings, attention
            weights/biases, FFN weights/biases, layer norms, and output
            projection. Uses actual dtype itemsize (4 bytes for float32).

        Returns:
            Total bytes used by all learnable parameters.

        Spec: SCENARIO-EBT-003
        """
        total = 0
        itemsize = self.token_embeddings.dtype.itemsize

        # Embeddings
        total += self.token_embeddings.size * itemsize
        total += self.pos_embeddings.size * itemsize

        # Transformer layers
        for layer in self.layers:
            # Attention weights and biases (Q, K, V, O)
            for w in [layer.w_q, layer.w_k, layer.w_v, layer.w_o]:
                total += w.size * itemsize
            for b in [layer.b_q, layer.b_k, layer.b_v, layer.b_o]:
                total += b.size * itemsize
            # Layer norm 1
            total += layer.ln1_gamma.size * itemsize
            total += layer.ln1_beta.size * itemsize
            # FFN weights and biases
            total += layer.w_ff1.size * itemsize
            total += layer.b_ff1.size * itemsize
            total += layer.w_ff2.size * itemsize
            total += layer.b_ff2.size * itemsize
            # Layer norm 2
            total += layer.ln2_gamma.size * itemsize
            total += layer.ln2_beta.size * itemsize

        # Final layer norm
        total += self.final_ln_gamma.size * itemsize
        total += self.final_ln_beta.size * itemsize

        # Output projection
        total += self.output_weight.size * itemsize
        total += itemsize  # output_bias (scalar)

        return total
