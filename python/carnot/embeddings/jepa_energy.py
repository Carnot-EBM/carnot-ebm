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


def embedding_repair(
    context_emb: jax.Array,
    prediction_emb: jax.Array,
    energy_model: ContextPredictionEnergy,
    steps: int = 50,
    step_size: float = 0.01,
) -> jax.Array:
    """Repair a prediction embedding by gradient descent on the JEPA energy.

    **Researcher summary:**
        Given a (context, prediction) embedding pair with high energy (meaning
        the prediction is incoherent with the context), descend on the energy
        surface by adjusting only the prediction embedding. After ``steps``
        iterations the repaired prediction should have lower energy — i.e., it
        is a more coherent continuation of the context.

    **Detailed explanation for engineers:**
        This is the bridge between "scoring" and "fixing." The JEPA energy
        function tells us *how bad* a prediction is (high energy = bad). But
        scoring alone is passive. ``embedding_repair`` makes it active: it
        takes a bad prediction embedding and *improves* it by walking downhill
        on the energy landscape while keeping the context embedding fixed.

        The algorithm is simple gradient descent:
        1. Concatenate context_emb and prediction_emb into a joint vector.
        2. Compute the energy E(ctx, pred).
        3. Compute dE/d(pred_emb) — the gradient of energy with respect to
           only the prediction half of the input.
        4. Update: pred_emb = pred_emb - step_size * dE/d(pred_emb).
        5. Repeat for ``steps`` iterations.

        **Why only update the prediction?**
        The context is the "ground truth" first half of the code. We don't want
        to change it. We only want to nudge the prediction embedding toward
        something that the energy model considers a coherent continuation.

        **How we get the prediction-only gradient:**
        We define a helper that takes pred_emb as input, concatenates it with
        the fixed context_emb, and returns the energy. Then ``jax.grad`` of
        this helper gives us dE/d(pred_emb) directly.

    Args:
        context_emb: 1-D JAX array of shape (embed_dim,) — the context
            embedding (held fixed throughout repair).
        prediction_emb: 1-D JAX array of shape (embed_dim,) — the initial
            (possibly bad) prediction embedding to be repaired.
        energy_model: A trained ContextPredictionEnergy model that scores
            (context, prediction) coherence.
        steps: Number of gradient descent iterations. More steps = more
            repair but diminishing returns. Default 50.
        step_size: Learning rate for gradient descent. Larger = faster
            convergence but risk of overshooting. Default 0.01.

    Returns:
        The repaired prediction embedding, same shape as ``prediction_emb``.

    Spec: REQ-JEPA-002, SCENARIO-JEPA-006
    """

    def _energy_of_pred(pred: jax.Array) -> jax.Array:
        """Energy as a function of prediction embedding only (context is fixed).

        **For engineers:**
            This closure captures ``context_emb`` and ``energy_model`` from the
            outer scope. JAX's grad will differentiate through the concatenation
            and the energy network, giving us the gradient with respect to
            ``pred`` only.
        """
        return energy_model.energy(jnp.concatenate([context_emb, pred]))

    pred = prediction_emb
    for _ in range(steps):
        grad = jax.grad(_energy_of_pred)(pred)
        pred = pred - step_size * grad

    return pred


def _corpus_entry_to_features(entry: dict) -> "jax.Array":
    """Convert a FOVERCorpusEntry-compatible dict to a 4-D feature vector.

    **For engineers:**
        Each FOVER corpus entry has a ``constraint_types`` list of Z3 labels per
        CoT step.  We summarise the distribution into four scalars:
            [frac_correct, frac_incorrect, frac_not_verifiable, normalized_n_steps]

        ``normalized_n_steps`` is clipped to [0, 1] using max 20 steps as the
        reference (any response with >= 20 steps gets value 1.0).  This captures
        response length as a proxy for complexity without unbounded magnitude.

        Empty constraint_types produce all-zero features, which is fine — the
        model learns to treat zero-feature entries as uninformative.
    """
    ctypes = entry.get("constraint_types", [])
    n = len(ctypes) if ctypes else 0
    if n == 0:
        return jnp.zeros(4)
    frac_correct = sum(1 for t in ctypes if t == "correct") / n
    frac_incorrect = sum(1 for t in ctypes if t == "incorrect") / n
    frac_nv = sum(1 for t in ctypes if t == "not_verifiable") / n
    norm_n = min(1.0, n / 20.0)
    return jnp.array([frac_correct, frac_incorrect, frac_nv, norm_n], dtype=jnp.float32)


def _leworldmodel_init_params(key: "jax.Array") -> dict:
    """Initialise LeWorldModel predictor parameters.

    **For engineers:**
        The predictor is a 2-layer MLP with two output heads:
        - mu_head: predicts the mean of the latent (used as the correctness score)
        - logvar_head: predicts log-variance (used for the KL regularisation term)

        Architecture: input(4) -> hidden(16) -> [mu(1), logvar(1)]

        Xavier uniform init for all weight matrices, zero biases.
        The logvar_bias is initialised to log(1) = 0, meaning the prior variance
        starts at 1.0 (exactly N(0,1)), so the KL starts near zero.
    """
    k1, k2, k3, k4 = jrandom.split(key, 4)
    in_dim, hid_dim, out_dim = 4, 16, 1
    lim1 = jnp.sqrt(6.0 / (in_dim + hid_dim))
    lim2 = jnp.sqrt(6.0 / (hid_dim + out_dim))
    return {
        "w1": jrandom.uniform(k1, (hid_dim, in_dim), minval=-lim1, maxval=lim1),
        "b1": jnp.zeros(hid_dim),
        "w_mu": jrandom.uniform(k2, (out_dim, hid_dim), minval=-lim2, maxval=lim2),
        "b_mu": jnp.zeros(out_dim),
        "w_lv": jrandom.uniform(k3, (out_dim, hid_dim), minval=-lim2, maxval=lim2),
        "b_lv": jnp.zeros(out_dim),
    }


def _leworldmodel_forward(
    params: dict, x: "jax.Array"
) -> tuple["jax.Array", "jax.Array"]:
    """Run the LeWorldModel predictor forward pass: features -> (mu, log_var).

    **For engineers:**
        Returns the mean ``mu`` and the log-variance ``log_var`` of the latent
        distribution q(z) = N(mu, exp(log_var)).  The mean is used as the
        predicted embedding; the log_var feeds the KL regularisation term.

        SiLU (Swish) activation in the hidden layer — smooth gradient flow.
    """
    h = jax.nn.silu(params["w1"] @ x + params["b1"])
    mu = params["w_mu"] @ h + params["b_mu"]
    log_var = params["w_lv"] @ h + params["b_lv"]
    return mu, log_var


def _leworldmodel_loss(
    params: dict,
    x: "jax.Array",
    y: "jax.Array",
    lambda_kl: float,
) -> tuple["jax.Array", "jax.Array", "jax.Array"]:
    """Compute L_total, L_pred, L_kl for a single example.

    **For engineers:**
        L_prediction = MSE(sigmoid(mu), y)  — mean-squared error of the sigmoid-scaled
            prediction vs the is_correct float target.  Sigmoid maps mu from R -> (0,1).
        L_kl = 0.5 * sum(exp(log_var) + mu^2 - 1 - log_var)  — KL(N(mu,exp(lv))||N(0,I)).
        L_total = L_pred + lambda_kl * L_kl

        KL is always >= 0 by Gibbs' inequality.  It equals 0 iff mu=0 and log_var=0,
        i.e. the latent is exactly N(0,1).  The lambda_kl weight (0.01) keeps the KL
        term one order of magnitude smaller than the MSE term at initialisation.
    """
    mu, log_var = _leworldmodel_forward(params, x)
    pred_emb = jax.nn.sigmoid(mu)
    l_pred = jnp.mean((pred_emb - y) ** 2)
    l_kl = 0.5 * jnp.sum(jnp.exp(log_var) + mu ** 2 - 1.0 - log_var)
    l_total = l_pred + lambda_kl * l_kl
    return l_total, l_pred, l_kl


def _auc_from_scores(scores: list[float], labels: list[float]) -> float:
    """Compute ROC-AUC without sklearn — trapezoid rule over sorted thresholds.

    **For engineers:**
        A clean manual AUC implementation to avoid the sklearn dependency in this
        low-level module.  We sort by descending score, walk thresholds, and accumulate
        TPR/FPR points for the trapezoidal rule.  Returns 0.5 when all labels are the
        same class (degenerate case — no discrimination possible).
    """
    n = len(scores)
    if n == 0:
        return 0.5
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pairs = sorted(zip(scores, labels), key=lambda t: -t[0])
    tp = fp = 0
    prev_tpr = prev_fpr = 0.0
    auc_val = 0.0
    for _, lbl in pairs:
        if lbl:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc_val += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_tpr, prev_fpr = tpr, fpr
    return auc_val


def train_leworldmodel(
    pairs: list[dict],
    lambda_kl: float = 0.01,
) -> list[tuple[int, float, float, float, float]]:
    """Train a JEPA predictor on FOVER corpus entries using the LeWorldModel two-term objective.

    **Researcher summary (RETRO-056 fix):**
        Standard BCE training collapsed on Exp 543 (AUC=0.444) because the corpus had
        only 24 pairs with 88% carry violations.  The LeWorldModel objective adds
        Gaussian KL regularisation that prevents latent collapse even when the training
        signal is sparse or skewed.  With >=100 diverse pairs (entropy >=1.0 bits)
        and the KL term, the predictor should recover to AUC >0.5 and ideally >0.8.

    **What this function does:**
        1. Converts each FOVERCorpusEntry-compatible dict to a 4-D feature vector
           [frac_correct, frac_incorrect, frac_not_verifiable, norm_n_steps].
        2. Initialises a 2-layer MLP with (mu, log_var) output heads.
        3. Trains for 200 epochs with AdamW (weight_decay=1e-4) on the two-term loss:
               L_total = L_prediction + lambda_kl * L_kl
           where L_prediction = MSE(sigmoid(mu), is_correct_float)
           and   L_kl = KL(N(mu, exp(log_var)) || N(0,I)).
        4. After each epoch, computes ROC-AUC on the full training set.

    **Why AdamW over Adam:**
        AdamW decouples the weight decay from the adaptive gradient scaling, which
        prevents the L2 regularisation from being swamped by large gradient magnitudes.
        On small datasets (< 200 pairs) this gives more stable convergence than Adam.

    Args:
        pairs: List of dicts with keys ``constraint_types`` (list[str]) and
               ``is_correct`` (bool).  Compatible with FOVERCorpusEntry dicts
               loaded directly from fover_corpus_v2.json.
        lambda_kl: Weight on the KL regularisation term.  Default 0.01 matches the
                   LeWorldModel paper recommendation for small models.

    Returns:
        training_history: List of length 200 (one entry per epoch).
        Each entry is a tuple: (epoch_int, total_loss, pred_loss, kl_loss, auc_float).
        Epoch indexing starts at 0.

    Spec: REQ-LEARN-047,
          SCENARIO-LEARN-076, SCENARIO-LEARN-077, SCENARIO-LEARN-078
    """
    import optax  # imported here to keep the module top-level import-light

    if not pairs:
        return [(i, 0.0, 0.0, 0.0, 0.5) for i in range(200)]

    # Build feature matrix and label vector from corpus entries
    features = [_corpus_entry_to_features(p) for p in pairs]
    labels = [float(bool(p.get("is_correct", False))) for p in pairs]
    X = jnp.stack(features)             # (n, 4)
    y = jnp.array(labels, dtype=jnp.float32)  # (n,)

    key = jrandom.PRNGKey(557)
    params = _leworldmodel_init_params(key)

    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    def _batch_loss_fn(p: dict) -> tuple["jax.Array", tuple["jax.Array", "jax.Array"]]:
        """Mean total loss + mean component losses over all training pairs."""
        def _single(xi: "jax.Array", yi: "jax.Array") -> tuple:
            return _leworldmodel_loss(p, xi, yi, lambda_kl)
        totals, preds, kls = jax.vmap(_single)(X, y)
        return jnp.mean(totals), (jnp.mean(preds), jnp.mean(kls))

    history: list[tuple[int, float, float, float, float]] = []

    for epoch in range(200):
        # Compute gradients w.r.t. params (first output of _batch_loss_fn)
        (total, (pred_l, kl_l)), grads = jax.value_and_grad(
            _batch_loss_fn, has_aux=True
        )(params)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Compute AUC on the full set (cheap at n<=200)
        mu_vals = jax.vmap(lambda xi: _leworldmodel_forward(params, xi)[0])(X)
        scores = [float(jax.nn.sigmoid(m)[0]) for m in mu_vals]
        auc = _auc_from_scores(scores, labels)

        history.append((epoch, float(total), float(pred_l), float(kl_l), auc))

    return history


def nearest_code_match(
    repaired_emb: jax.Array,
    codebook_embs: jax.Array,
    codebook_texts: list[str],
) -> str:
    """Find the codebook entry whose embedding is closest to the repaired embedding.

    **Researcher summary:**
        Cosine similarity search over a codebook of (embedding, code_text) pairs.
        Returns the code text whose embedding has the highest cosine similarity
        to ``repaired_emb``.

    **Detailed explanation for engineers:**
        After ``embedding_repair`` produces a repaired prediction embedding, it
        lives in continuous vector space — it doesn't correspond to any actual
        code snippet. To get back to real code, we need to find the *nearest
        neighbor* in a codebook of known embeddings.

        **Why cosine similarity instead of Euclidean distance?**
        Embeddings from neural networks often vary in magnitude depending on
        input length or content. Cosine similarity measures the *angle* between
        vectors, ignoring magnitude, which makes it more robust for comparing
        embeddings of different-length code snippets. Two embeddings pointing
        in the same direction (cosine ~ 1.0) represent similar code regardless
        of their norms.

        **The math:**
        cosine_sim(a, b) = (a · b) / (||a|| * ||b||)

        We compute this for every codebook entry and return the text associated
        with the highest similarity score.

        **Edge case — zero-norm vectors:**
        If either the repaired embedding or a codebook entry has zero norm
        (e.g., from an empty code snippet), the cosine similarity is undefined.
        We add a small epsilon (1e-8) to the denominator to avoid division by
        zero. Zero-norm entries will get near-zero similarity, which is the
        correct behavior (they shouldn't match anything).

    Args:
        repaired_emb: 1-D JAX array of shape (embed_dim,) — the repaired
            prediction embedding from ``embedding_repair``.
        codebook_embs: 2-D JAX array of shape (n_entries, embed_dim) — the
            embeddings of all known code snippets.
        codebook_texts: List of ``n_entries`` code strings, one per codebook
            embedding. Must be the same length as ``codebook_embs``.

    Returns:
        The code text string from ``codebook_texts`` whose embedding is most
        similar to ``repaired_emb``.

    Spec: REQ-JEPA-002, SCENARIO-JEPA-007
    """
    # Dot product of repaired_emb with every codebook entry
    dots = codebook_embs @ repaired_emb  # shape (n_entries,)

    # Norms for cosine similarity (epsilon prevents division by zero)
    eps = 1e-8
    repaired_norm = jnp.linalg.norm(repaired_emb) + eps
    codebook_norms = jnp.linalg.norm(codebook_embs, axis=1) + eps

    cosine_similarities = dots / (codebook_norms * repaired_norm)

    best_idx = int(jnp.argmax(cosine_similarities))
    return codebook_texts[best_idx]
