"""EB-JEPA context-prediction energy function and training data generation.

**Researcher summary:**
    Implements an energy function for EB-JEPA (Energy-Based Joint Embedding
    Predictive Architecture). Takes concatenated (context_embedding, prediction_embedding)
    and outputs a scalar energy: low if the prediction is a coherent continuation of
    the context, high otherwise. Trained with Noise Contrastive Estimation (NCE).

**Detailed explanation for engineers:**
    EB-JEPA is inspired by Yann LeCun's Joint Embedding Predictive Architecture
    (JEPA), which predicts missing context in *embedding space* rather than in
    pixel/token space. The key idea is:

    1. Take a piece of content (e.g., a Python function) and split it into two
       halves: a "context" (first half) and a "prediction target" (second half).

    2. Embed each half independently into a dense vector using an embedding
       function (here we use AST-based code embeddings from
       ``carnot.verify.python_types.ast_code_to_embedding``).

    3. Concatenate the two embeddings and feed them into an energy function
       (a Gibbs-like neural network that outputs a scalar).

    4. **Training objective**: The energy function should assign LOW energy to
       correct (context, continuation) pairs (real first+second halves of the
       same function) and HIGH energy to incorrect pairs (first half of function
       A paired with second half of function B).

    **Noise Contrastive Estimation (NCE):**
    NCE is a training technique that avoids computing the intractable partition
    function of an energy-based model. Instead of maximizing the likelihood
    directly, NCE trains the model to distinguish between "data" samples
    (correct pairs) and "noise" samples (shuffled/incorrect pairs).

    The NCE loss for a single data point is:
        L = -log(sigmoid(-E(correct))) - log(sigmoid(E(noise)))

    This pushes correct pair energies down and noise pair energies up.
    Over many batches, the energy function learns to recognize coherent
    context-prediction relationships.

    **Why this matters for autonomous code generation:**
    If an LLM generates the second half of a function, we can use this energy
    function to score whether that continuation is coherent with the first half.
    Low energy = likely correct continuation. High energy = probably wrong.
    This gives us a differentiable "code coherence" signal that doesn't require
    executing the code.

Spec: REQ-JEPA-001
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import AutoGradMixin
from carnot.models.gibbs import GibbsConfig, _apply_activation


@dataclass
class JEPAEnergyConfig:
    """Configuration for the EB-JEPA context-prediction energy function.

    **Researcher summary:**
        Specifies the embedding dimension, network topology, and activation
        for the JEPA energy scorer. Input dim = 2 * embed_dim (concatenation
        of context and prediction embeddings).

    **Detailed explanation for engineers:**
        The energy function receives a concatenation of two embedding vectors:
        one for the "context" (e.g., first half of a function) and one for the
        "prediction" (e.g., second half). So if each embedding is 64-dimensional,
        the network input is 128-dimensional.

        The hidden layers progressively compress this joint representation
        down to a scalar energy value. The default architecture
        (embed_dim=64 -> hidden [64, 32] -> scalar) is compact enough for
        fast training but expressive enough to capture context-prediction
        relationships in code embeddings.

    Attributes:
        embed_dim: Dimensionality of each individual embedding (context or
            prediction). The network input will be 2 * embed_dim.
        hidden_dims: Sizes of hidden layers in the energy network.
        activation: Nonlinear activation function ("silu", "relu", or "tanh").

    Spec: REQ-JEPA-001
    """

    embed_dim: int = 64
    hidden_dims: list[int] | None = None
    activation: str = "silu"

    def __post_init__(self) -> None:
        """Set default hidden_dims if not provided."""
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]

    def validate(self) -> None:
        """Validate configuration parameters.

        **For engineers:**
            Checks that embed_dim is positive, hidden_dims is non-empty with
            all positive values, and activation is one of the supported options.

        Raises:
            ValueError: If any parameter is invalid.

        Spec: SCENARIO-JEPA-001
        """
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if not self.hidden_dims or len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must have at least one layer")
        if any(d <= 0 for d in self.hidden_dims):
            raise ValueError("all hidden_dims must be > 0")
        if self.activation not in ("silu", "relu", "tanh"):
            raise ValueError(
                f"Unknown activation: {self.activation}. Use 'silu', 'relu', or 'tanh'."
            )


class ContextPredictionEnergy(AutoGradMixin):
    """EB-JEPA energy function: scores coherence of (context, prediction) pairs.

    **Researcher summary:**
        E(concat(ctx_emb, pred_emb)) -> scalar. Low energy = coherent continuation.
        Gibbs-like MLP architecture. Gradients via jax.grad (AutoGradMixin).

    **Detailed explanation for engineers:**
        This is structurally identical to a GibbsModel, but with a specific
        semantic purpose: it takes the concatenation of two embeddings (context
        and prediction) as input and outputs a scalar energy that measures how
        well the prediction "continues" the context.

        The forward pass is:
        1. Receive input x of shape (2 * embed_dim,) — the concatenation of
           context_embedding and prediction_embedding.
        2. Pass through hidden layers: h = activation(W @ h + b)
        3. Linear readout to scalar: E = w_out @ h_last + b_out

        Because it inherits AutoGradMixin, ``grad_energy`` and ``energy_batch``
        are automatically available via jax.grad and jax.vmap.

        **Usage pattern:**
        ```python
        model = ContextPredictionEnergy(JEPAEnergyConfig(embed_dim=64))
        ctx_emb = jnp.ones(64)    # embedding of first half of code
        pred_emb = jnp.ones(64)   # embedding of second half of code
        pair = jnp.concatenate([ctx_emb, pred_emb])  # shape (128,)
        energy = model.energy(pair)  # scalar — lower = more coherent
        ```

    Spec: REQ-JEPA-001
    """

    def __init__(
        self,
        config: JEPAEnergyConfig,
        key: jax.Array | None = None,
    ) -> None:
        """Create a new ContextPredictionEnergy model with initialized parameters.

        **Detailed explanation for engineers:**
            Validates the config and builds the network layer by layer, exactly
            like GibbsModel. The input dimension is 2 * embed_dim because the
            input is the concatenation of context and prediction embeddings.

            Uses Xavier/Glorot uniform initialization for weight matrices
            (same rationale as GibbsModel — keeps activation variance stable
            across layers). Biases are zero-initialized. Output layer starts
            at zero so initial energy is 0 for all inputs.

        Args:
            config: A JEPAEnergyConfig specifying embedding dimension and
                network architecture.
            key: JAX PRNG key for random initialization. If None, uses seed 0.

        Raises:
            ValueError: If config has invalid values.

        Spec: REQ-JEPA-001, SCENARIO-JEPA-001
        """
        config.validate()
        self.config = config

        if key is None:
            key = jrandom.PRNGKey(0)

        hidden_dims = config.hidden_dims
        assert hidden_dims is not None  # validated above

        # Input is the concatenation of context_emb and prediction_emb
        input_dim = 2 * config.embed_dim

        # Build hidden layers: each transforms prev_dim -> hidden_dim
        self.layers: list[tuple[jax.Array, jax.Array]] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            k_w, key = jrandom.split(key)
            # Xavier uniform initialization: scale by sqrt(6 / (fan_in + fan_out))
            limit = jnp.sqrt(6.0 / (prev_dim + hidden_dim))
            weight = jrandom.uniform(
                k_w, (hidden_dim, prev_dim), minval=-limit, maxval=limit
            )
            bias = jnp.zeros(hidden_dim)
            self.layers.append((weight, bias))
            prev_dim = hidden_dim

        # Output layer: dot product from last hidden dim -> scalar energy.
        # Zero-initialized so model starts with energy = 0 for all inputs.
        self.output_weight = jnp.zeros(prev_dim)
        self.output_bias = 0.0

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute scalar energy E(x) for a concatenated (context, prediction) pair.

        **Researcher summary:**
            Forward pass: concat(ctx, pred) -> dense layers with activation -> scalar.

        **Detailed explanation for engineers:**
            The input x should be the concatenation of a context embedding and
            a prediction embedding, giving shape (2 * embed_dim,). The network
            processes this through hidden layers with the configured activation
            function, then produces a scalar output via a linear readout.

            Low energy means the context and prediction are coherent (they
            likely came from the same original content). High energy means
            they are incoherent (likely from different sources).

        Args:
            x: A 1-D JAX array of shape (2 * embed_dim,) — the concatenation
                of context_embedding and prediction_embedding.

        Returns:
            A scalar JAX array representing the coherence energy.

        Spec: REQ-JEPA-001, SCENARIO-JEPA-002
        """
        h = x
        for weight, bias in self.layers:
            h = _apply_activation(weight @ h + bias, self.config.activation)

        return self.output_weight @ h + self.output_bias

    @property
    def input_dim(self) -> int:
        """Number of input dimensions (2 * embed_dim)."""
        return 2 * self.config.embed_dim

    def energy_pair(
        self, context_emb: jax.Array, prediction_emb: jax.Array
    ) -> jax.Array:
        """Convenience method: compute energy from separate context and prediction embeddings.

        **For engineers:**
            Instead of manually concatenating the embeddings before calling
            ``energy()``, you can pass them separately. This method concatenates
            them and calls ``energy()`` for you.

        Args:
            context_emb: 1-D JAX array of shape (embed_dim,).
            prediction_emb: 1-D JAX array of shape (embed_dim,).

        Returns:
            Scalar energy.

        Spec: REQ-JEPA-001
        """
        return self.energy(jnp.concatenate([context_emb, prediction_emb]))


def nce_loss(
    model: ContextPredictionEnergy,
    data_pairs: jax.Array,
    noise_pairs: jax.Array,
) -> jax.Array:
    """Compute Noise Contrastive Estimation loss for JEPA energy training.

    **Researcher summary:**
        NCE loss = -mean(log(sigmoid(-E(data)))) - mean(log(sigmoid(E(noise)))).
        Pushes data energies down and noise energies up.

    **Detailed explanation for engineers:**
        Noise Contrastive Estimation (NCE) is a technique for training
        energy-based models without computing the intractable partition function
        (the normalizing constant Z = sum of exp(-E(x)) over all possible x).

        The key insight: instead of modeling the full probability distribution,
        train the model to *distinguish* between real data and noise. This is
        a binary classification problem:
        - Real data pairs (correct context + correct prediction) should have
          LOW energy, so sigmoid(-E) should be close to 1.
        - Noise pairs (mismatched context + prediction) should have HIGH energy,
          so sigmoid(E) should be close to 1.

        The loss combines both terms:
        - Data term: -mean(log(sigmoid(-E(data))))
          Penalizes high energy on correct pairs.
        - Noise term: -mean(log(sigmoid(E(noise))))
          Penalizes low energy on incorrect pairs.

        We add a small epsilon (1e-7) inside the log to prevent log(0) = -inf
        when the sigmoid outputs are exactly 0 or 1.

    Args:
        model: The ContextPredictionEnergy model to evaluate.
        data_pairs: 2-D array of shape (n_data, 2*embed_dim) — correct pairs.
        noise_pairs: 2-D array of shape (n_noise, 2*embed_dim) — shuffled pairs.

    Returns:
        Scalar NCE loss value.

    Spec: REQ-JEPA-001, SCENARIO-JEPA-003
    """
    eps = 1e-7

    # Compute energies for all data and noise pairs in batch
    data_energies = model.energy_batch(data_pairs)
    noise_energies = model.energy_batch(noise_pairs)

    # Data term: correct pairs should have low energy -> sigmoid(-E) ~ 1
    data_term = -jnp.mean(jnp.log(jax.nn.sigmoid(-data_energies) + eps))

    # Noise term: incorrect pairs should have high energy -> sigmoid(E) ~ 1
    noise_term = -jnp.mean(jnp.log(jax.nn.sigmoid(noise_energies) + eps))

    return data_term + noise_term


def generate_jepa_training_data(
    code_snippets: list[str],
    embed_dim: int = 64,
    key: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Generate training data for EB-JEPA by splitting code and creating correct/noise pairs.

    **Researcher summary:**
        Splits each code snippet at its midpoint (by lines), embeds each half
        via ``ast_code_to_embedding``, and produces correct pairs (same function)
        and noise pairs (shuffled second halves from different functions).

    **Detailed explanation for engineers:**
        This function creates the training data for the JEPA energy function:

        1. **Split each code snippet** into two halves by splitting at the
           midpoint line. For example, a 10-line function gets split into
           lines 1-5 (context) and lines 6-10 (prediction target).

        2. **Embed each half** using ``ast_code_to_embedding`` from the
           verify module. This produces a fixed-size feature vector capturing
           the AST structure of each code half.

        3. **Create correct pairs**: For each snippet, concatenate the context
           embedding with the prediction embedding from the SAME snippet.
           These are the "data" samples — coherent continuations.

        4. **Create noise pairs**: Shuffle the prediction embeddings (second
           halves) so that each context gets paired with a prediction from
           a DIFFERENT snippet. These are the "noise" samples — incoherent
           continuations.

        The shuffling is done via a random permutation using the provided
        JAX PRNG key, ensuring reproducibility.

        **Why line-based splitting?**
        Code has natural line boundaries (statements, definitions). Splitting
        at the midpoint line keeps both halves syntactically meaningful —
        each half is likely to contain complete statements rather than
        being cut mid-expression.

        **Edge cases:**
        - Single-line snippets: context = the line, prediction = empty string.
          The AST embedding of an empty string is a zero vector, which is
          fine — it just means "no structure."
        - Snippets that are too short to embed meaningfully will produce
          near-zero embeddings, which the energy function will learn to
          handle during training.

    Args:
        code_snippets: List of Python source code strings (functions, classes,
            or any multi-line code). Should have at least 2 snippets to create
            meaningful noise pairs.
        embed_dim: Dimensionality for the AST embeddings. Must match the
            JEPAEnergyConfig.embed_dim used for the energy model.
        key: JAX PRNG key for shuffling noise pairs. If None, uses seed 42.

    Returns:
        A tuple (data_pairs, noise_pairs) where:
        - data_pairs: shape (n_snippets, 2*embed_dim) — correct context+prediction
        - noise_pairs: shape (n_snippets, 2*embed_dim) — shuffled context+prediction

    Spec: REQ-JEPA-001, SCENARIO-JEPA-004
    """
    from carnot.verify.python_types import ast_code_to_embedding

    if key is None:
        key = jrandom.PRNGKey(42)

    context_embeddings = []
    prediction_embeddings = []

    for code in code_snippets:
        lines = code.split("\n")
        mid = max(1, len(lines) // 2)
        context_code = "\n".join(lines[:mid])
        prediction_code = "\n".join(lines[mid:])

        ctx_emb = ast_code_to_embedding(textwrap.dedent(context_code), feature_dim=embed_dim)
        pred_emb = ast_code_to_embedding(textwrap.dedent(prediction_code), feature_dim=embed_dim)

        context_embeddings.append(ctx_emb)
        prediction_embeddings.append(pred_emb)

    ctx_stack = jnp.stack(context_embeddings)    # (n, embed_dim)
    pred_stack = jnp.stack(prediction_embeddings)  # (n, embed_dim)

    # Correct pairs: context[i] + prediction[i] from the same snippet
    data_pairs = jnp.concatenate([ctx_stack, pred_stack], axis=1)  # (n, 2*embed_dim)

    # Noise pairs: roll prediction embeddings by 1 position so that context[i]
    # pairs with prediction[(i+1) % n]. A circular shift guarantees that NO
    # element stays in its original position (unlike a random permutation which
    # can return the identity for small n). This is a simple derangement.
    shuffled_pred = jnp.roll(pred_stack, shift=1, axis=0)
    noise_pairs = jnp.concatenate([ctx_stack, shuffled_pred], axis=1)

    return data_pairs, noise_pairs


def train_jepa_energy(
    model: ContextPredictionEnergy,
    data_pairs: jax.Array,
    noise_pairs: jax.Array,
    learning_rate: float = 0.01,
    n_steps: int = 100,
    key: jax.Array | None = None,
) -> list[float]:
    """Train the JEPA energy function using NCE on the given data/noise pairs.

    **Researcher summary:**
        Gradient descent on NCE loss. Updates all model parameters (hidden
        layer weights/biases + output weight/bias) using vanilla SGD.
        Returns loss history for monitoring convergence.

    **Detailed explanation for engineers:**
        This is a simple training loop that:

        1. Computes the NCE loss (see ``nce_loss`` above).
        2. Computes gradients of the loss with respect to ALL model parameters
           using ``jax.grad``. Since the model stores parameters as instance
           attributes (not in a JAX pytree), we extract them into lists,
           compute gradients via a closure, and update in-place.
        3. Updates parameters via vanilla SGD: param -= lr * grad.
        4. Records the loss at each step for monitoring.

        **Why vanilla SGD instead of Adam?**
        For this small network and training setup, SGD is sufficient and has
        fewer moving parts. In production, you would likely use Optax with
        Adam or similar. This keeps the implementation simple and dependency-light.

        **Why extract parameters?**
        JAX's ``grad`` works on functions of arrays, not on object methods
        directly. We extract the model's parameters into flat lists, define
        a pure function that computes the loss given those parameters, and
        use ``jax.grad`` on that pure function. After getting gradients, we
        write the updated parameters back into the model.

    Args:
        model: The ContextPredictionEnergy model to train.
        data_pairs: Correct (context, prediction) pairs — shape (n, 2*embed_dim).
        noise_pairs: Shuffled noise pairs — shape (n, 2*embed_dim).
        learning_rate: SGD step size. Default 0.01.
        n_steps: Number of gradient descent steps. Default 100.
        key: Unused, reserved for future mini-batch sampling.

    Returns:
        List of NCE loss values at each training step.

    Spec: REQ-JEPA-001, SCENARIO-JEPA-005
    """

    activation = model.config.activation

    def _energy_single(
        layer_weights: list[jax.Array],
        layer_biases: list[jax.Array],
        output_weight: jax.Array,
        output_bias: jax.Array,
        x: jax.Array,
    ) -> jax.Array:
        """Pure functional energy computation for a single input.

        **For engineers:**
            JAX's grad requires a pure function — no side effects, no object
            mutation. This takes all parameters as explicit arguments so JAX
            can trace through the computation graph and compute gradients.
            The model's config (activation function) is captured from the
            outer scope as a constant.
        """
        h = x
        for w, b in zip(layer_weights, layer_biases):
            h = _apply_activation(w @ h + b, activation)
        return output_weight @ h + output_bias

    def _energy_batch_fn(
        layer_weights: list[jax.Array],
        layer_biases: list[jax.Array],
        output_weight: jax.Array,
        output_bias: jax.Array,
        xs: jax.Array,
    ) -> jax.Array:
        """Batched energy using vmap over the pure functional version."""
        return jax.vmap(
            lambda x: _energy_single(
                layer_weights, layer_biases, output_weight, output_bias, x
            )
        )(xs)

    def _loss_fn(
        layer_weights: list[jax.Array],
        layer_biases: list[jax.Array],
        output_weight: jax.Array,
        output_bias: jax.Array,
    ) -> jax.Array:
        """Pure NCE loss as a function of explicit parameters."""
        eps = 1e-7
        data_energies = _energy_batch_fn(
            layer_weights, layer_biases, output_weight, output_bias, data_pairs
        )
        noise_energies = _energy_batch_fn(
            layer_weights, layer_biases, output_weight, output_bias, noise_pairs
        )
        data_term = -jnp.mean(jnp.log(jax.nn.sigmoid(-data_energies) + eps))
        noise_term = -jnp.mean(jnp.log(jax.nn.sigmoid(noise_energies) + eps))
        return data_term + noise_term

    loss_history: list[float] = []

    for _step in range(n_steps):
        # Extract current parameters as separate arrays for jax.grad
        weights = [w for w, _b in model.layers]
        biases = [b for _w, b in model.layers]

        # Compute loss and gradients with respect to all parameters
        loss_val, grads = jax.value_and_grad(_loss_fn, argnums=(0, 1, 2, 3))(
            weights, biases, model.output_weight, jnp.array(model.output_bias)
        )

        grad_weights, grad_biases, grad_ow, grad_ob = grads

        # SGD update: param -= learning_rate * gradient
        new_layers = []
        for i in range(len(model.layers)):
            new_w = weights[i] - learning_rate * grad_weights[i]
            new_b = biases[i] - learning_rate * grad_biases[i]
            new_layers.append((new_w, new_b))
        model.layers = new_layers

        model.output_weight = model.output_weight - learning_rate * grad_ow
        model.output_bias = float(jnp.array(model.output_bias) - learning_rate * grad_ob)

        loss_history.append(float(loss_val))

    return loss_history
