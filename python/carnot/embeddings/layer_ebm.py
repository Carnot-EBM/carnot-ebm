"""Layer-selective EBM for hallucination monitoring.

**Researcher summary:**
    Not all transformer layers are equally informative for detecting
    hallucinations. This module identifies the 2-3 most discriminative
    layers via the Fisher criterion, trains a compact Gibbs model on
    their concatenated activations using NCE, and wraps everything in
    a ``LayerEBMVerifier`` that combines extraction + direction-finding
    + trained energy model into one callable.

**Detailed explanation for engineers:**
    When monitoring an LLM for hallucinations, extracting activations
    from *every* layer is expensive (memory and compute). Most layers
    carry redundant or uninformative signal for the binary question
    "is this hallucinating?" This module solves the problem in three
    steps:

    1. **Identify critical layers** (``identify_critical_layers``):
       Given per-layer activations from a set of known-correct and
       known-hallucinated examples, compute a Fisher discriminant
       ratio for each layer. The Fisher criterion measures how well
       the means of the two classes are separated relative to their
       within-class scatter:

           F(layer) = ||mean_halluc - mean_correct||^2
                      / (var_correct + var_halluc + eps)

       Layers with high F scores carry the strongest signal. We
       return the top-k layer indices (default 3).

    2. **Train a layer-selective EBM** (``train_layer_ebm``):
       Concatenate the activations from only the critical layers
       (reducing dimensionality vs. using all layers), then train
       a ``GibbsModel`` using Noise Contrastive Estimation (NCE).
       The training loop mirrors ``train_sat_verifier`` from the
       inference module: extract parameters as a pytree, compute
       gradients with ``jax.grad``, and apply vanilla SGD.

    3. **End-to-end verifier** (``LayerEBMVerifier``):
       Combines the hallucination direction (from
       ``find_hallucination_direction``) with the trained layer EBM
       into a single object. Given per-layer activations, it:
       (a) selects the critical layers,
       (b) concatenates their activations,
       (c) computes the trained EBM energy on the concatenation,
       (d) computes the hallucination direction energy,
       (e) returns the sum as the total hallucination energy.

       This sum-of-energies design composes naturally with Carnot's
       ``ComposedEnergy`` and verify-and-repair pipeline.

Spec: REQ-INFER-015
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.embeddings.hallucination_direction import (
    find_hallucination_direction,
    hallucination_energy,
)
from carnot.inference.learned_verifier import LearnedVerifierConfig
from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.training.nce import nce_loss
from carnot.verify.constraint import BaseConstraint


@dataclass
class LayerEBMConfig:
    """Configuration for layer-selective EBM training.

    **Researcher summary:**
        Controls which layers to select (top_k), the Gibbs model
        architecture, and the NCE training hyperparameters.

    **Detailed explanation for engineers:**
        - ``top_k_layers``: How many critical layers to keep. More
          layers means more features for the EBM but also higher
          dimensionality and slower training. 3 is a good default
          for most transformer architectures (24-32 layers).
        - ``hidden_dims``: Architecture of the Gibbs model that
          scores concatenated activations. Kept small ([64, 32])
          because the input is already a compact subset of layers.
        - ``n_epochs``: Number of full passes over the training data.
        - ``learning_rate``: Step size for vanilla SGD.
        - ``seed``: PRNG seed for reproducibility.

    Spec: REQ-INFER-015
    """

    top_k_layers: int = 3
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    n_epochs: int = 100
    learning_rate: float = 0.01
    seed: int = 42

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is out of range.
        """
        if self.top_k_layers < 1:
            raise ValueError("top_k_layers must be >= 1")
        if not self.hidden_dims or any(d <= 0 for d in self.hidden_dims):
            raise ValueError("hidden_dims must be non-empty with all values > 0")
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")


def identify_critical_layers(
    activations_correct: dict[int, list[jax.Array]],
    activations_hallucinated: dict[int, list[jax.Array]],
    top_k: int = 3,
) -> list[int]:
    """Identify the most discriminative transformer layers for hallucination detection.

    **Researcher summary:**
        Computes the Fisher discriminant ratio per layer and returns
        the top-k layer indices ranked by discrimination power.

    **Detailed explanation for engineers:**
        The Fisher criterion (also called Fisher's linear discriminant
        ratio) measures how well two classes are separated in a given
        feature space. For each layer, we compute:

            F = ||mu_halluc - mu_correct||^2 / (sigma^2_correct + sigma^2_halluc + eps)

        where:
        - mu_correct, mu_halluc are the per-layer mean activation
          vectors (averaged over both samples and sequence positions).
        - sigma^2 is the mean variance of activations within each class.
        - eps prevents division by zero when activations are constant.

        **Why Fisher and not something fancier (mutual information,
        kernel methods)?**
        Fisher is cheap (O(n * d) per layer), interpretable, and works
        well when the hallucination signal is roughly linear in
        activation space — which is exactly the assumption behind
        ``find_hallucination_direction``. If the signal were highly
        nonlinear, you'd want a more expensive measure, but at that
        point the linear direction approach wouldn't work either.

        **Why mean over sequence positions?**
        Transformer activations have shape ``(seq_len, hidden_dim)``.
        Different tokens may carry different amounts of hallucination
        signal, but averaging gives a robust per-sample representation
        that's invariant to sequence length. This is the same pooling
        strategy used by ``compute_activation_stats``.

    Args:
        activations_correct: Dict mapping layer index to a list of
            activation arrays from correct outputs. Each array has
            shape ``(seq_len, hidden_dim)`` or ``(hidden_dim,)``.
        activations_hallucinated: Dict mapping layer index to a list
            of activation arrays from hallucinated outputs. Same shape.
        top_k: Number of top layers to return. Clamped to the number
            of available layers if larger.

    Returns:
        List of layer indices (ints) sorted by descending Fisher score,
        length ``min(top_k, n_layers)``.

    Raises:
        ValueError: If no common layers exist between the two dicts,
            or if either dict is empty.

    Spec: REQ-INFER-015
    """
    if not activations_correct or not activations_hallucinated:
        raise ValueError("Both activation dicts must be non-empty")

    # Find layers present in both dicts.
    common_layers = sorted(
        set(activations_correct.keys()) & set(activations_hallucinated.keys())
    )
    if not common_layers:
        raise ValueError("No common layers between correct and hallucinated activations")

    # Clamp top_k to available layers.
    top_k = min(top_k, len(common_layers))

    fisher_scores: dict[int, float] = {}

    for layer_idx in common_layers:
        correct_acts = activations_correct[layer_idx]
        halluc_acts = activations_hallucinated[layer_idx]

        if not correct_acts or not halluc_acts:
            fisher_scores[layer_idx] = 0.0
            continue

        # Pool each sample: if (seq_len, hidden_dim), take mean over seq_len.
        # If already (hidden_dim,), use as-is.
        def _pool(act: jax.Array) -> jax.Array:
            if act.ndim == 2:
                return jnp.mean(act, axis=0)
            return act

        correct_pooled = jnp.stack([_pool(a) for a in correct_acts])
        halluc_pooled = jnp.stack([_pool(a) for a in halluc_acts])

        # Centroids.
        mu_correct = jnp.mean(correct_pooled, axis=0)
        mu_halluc = jnp.mean(halluc_pooled, axis=0)

        # Between-class distance: ||mu_halluc - mu_correct||^2.
        between = float(jnp.sum((mu_halluc - mu_correct) ** 2))

        # Within-class scatter: mean variance per class.
        var_correct = float(jnp.mean(jnp.var(correct_pooled, axis=0)))
        var_halluc = float(jnp.mean(jnp.var(halluc_pooled, axis=0)))

        eps = 1e-8
        fisher_scores[layer_idx] = between / (var_correct + var_halluc + eps)

    # Sort by Fisher score descending, return top-k layer indices.
    ranked = sorted(fisher_scores, key=lambda k: fisher_scores[k], reverse=True)
    return ranked[:top_k]


def _concat_critical_activations(
    activations: dict[int, list[jax.Array]],
    critical_layers: list[int],
) -> jax.Array:
    """Concatenate pooled activations from critical layers into a flat matrix.

    **Detailed explanation for engineers:**
        For each sample, we take the activations from each critical layer,
        pool them (mean over sequence positions if 2-D), and concatenate
        them into a single vector. The result is a matrix of shape
        ``(n_samples, sum_of_hidden_dims)`` suitable for feeding into
        a Gibbs model.

    Args:
        activations: Dict mapping layer index to list of activation arrays.
        critical_layers: Which layer indices to use.

    Returns:
        2-D JAX array of shape ``(n_samples, total_dim)``.
    """
    n_samples = len(activations[critical_layers[0]])
    per_sample: list[jax.Array] = []

    for i in range(n_samples):
        parts: list[jax.Array] = []
        for layer_idx in critical_layers:
            act = activations[layer_idx][i]
            # Pool over sequence dimension if needed.
            if act.ndim == 2:
                act = jnp.mean(act, axis=0)
            parts.append(act)
        per_sample.append(jnp.concatenate(parts))

    return jnp.stack(per_sample)


def train_layer_ebm(
    correct_activations: jax.Array,
    hallucinated_activations: jax.Array,
    config: LearnedVerifierConfig | None = None,
) -> GibbsModel:
    """Train a Gibbs model to distinguish correct from hallucinated activations via NCE.

    **Researcher summary:**
        NCE training on concatenated critical-layer activations.
        Same paradigm as ``train_sat_verifier``: extract params as
        pytree, ``jax.grad`` for gradients, vanilla SGD updates.

    **Detailed explanation for engineers:**
        This function follows the exact same training pattern as
        ``train_sat_verifier`` from ``carnot.inference.learned_verifier``:

        1. Create a ``GibbsModel`` whose input dimension equals the
           total dimensionality of the concatenated critical-layer
           activations.
        2. Use the correct activations as "data" (should get low energy)
           and hallucinated activations as "noise" (should get high energy).
        3. Compute NCE loss, take gradients w.r.t. all model parameters,
           and apply vanilla SGD updates.

        After training, ``model.energy(x)`` should be low for activation
        patterns that look like correct outputs and high for patterns
        that look like hallucinations.

        **Why not use the same noise-generation approach as SAT?**
        In the SAT case, we need to generate satisfying assignments
        via rejection sampling. Here, we already have both classes of
        data (correct and hallucinated) — they come directly from
        running the LLM on known-correct and known-incorrect prompts.
        So we just use them directly as data and noise for NCE.

    Args:
        correct_activations: Concatenated critical-layer activations from
            correct outputs, shape ``(n_correct, total_dim)``.
        hallucinated_activations: Same for hallucinated outputs, shape
            ``(n_hallucinated, total_dim)``.
        config: Training hyperparameters. Uses defaults if None.

    Returns:
        Trained ``GibbsModel`` whose energy function scores hallucination
        likelihood.

    Spec: REQ-INFER-015
    """
    if config is None:
        config = LearnedVerifierConfig()

    input_dim = correct_activations.shape[1]

    key = jrandom.PRNGKey(config.seed)
    key, model_key = jrandom.split(key)

    gibbs_config = GibbsConfig(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        activation="silu",
    )
    model = GibbsModel(gibbs_config, key=model_key)

    # Pytree parameter extraction/injection — same pattern as train_sat_verifier.
    def get_params(m: GibbsModel) -> dict:  # type: ignore[type-arg]
        return {
            "layers": [(w, b) for w, b in m.layers],
            "output_weight": m.output_weight,
            "output_bias": m.output_bias,
        }

    def set_params(m: GibbsModel, params: dict) -> None:  # type: ignore[type-arg]
        m.layers = list(params["layers"])
        m.output_weight = params["output_weight"]
        m.output_bias = params["output_bias"]

    def loss_fn(params: dict) -> jax.Array:  # type: ignore[type-arg]
        old = get_params(model)
        set_params(model, params)
        result = nce_loss(model, correct_activations, hallucinated_activations)
        set_params(model, old)
        return result

    params = get_params(model)
    for _epoch in range(config.n_epochs):
        set_params(model, params)
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(
            lambda p, g: p - config.learning_rate * g,
            params,
            grads,
        )

    set_params(model, params)
    return model


class LayerEBMVerifier(BaseConstraint):
    """End-to-end hallucination verifier combining layer selection + trained EBM.

    **Researcher summary:**
        Given per-layer activations, selects critical layers, concatenates
        their activations, and computes a combined energy from the trained
        Gibbs model and the hallucination direction projection.

    **Detailed explanation for engineers:**
        This class ties together the full pipeline:

        1. **Critical layers**: A pre-computed list of the most
           informative layer indices (from ``identify_critical_layers``).
        2. **Trained EBM**: A ``GibbsModel`` trained on concatenated
           critical-layer activations via NCE.
        3. **Hallucination direction**: A direction vector (from
           ``find_hallucination_direction``) for linear projection.

        When you call ``energy(x)`` with the concatenated critical-layer
        activations, it returns the sum of:
        - The trained EBM's energy (nonlinear, learned from data).
        - The hallucination direction projection (linear baseline).

        This sum-of-energies approach means the verifier degrades
        gracefully: even if the trained model overfits, the linear
        direction still provides a floor of detection quality.

        ``LayerEBMVerifier`` implements ``BaseConstraint``, so it plugs
        directly into ``ComposedEnergy`` and the verify-and-repair loop.

    Attributes:
        critical_layers: Layer indices selected by Fisher criterion.
        model: Trained GibbsModel for scoring activations.
        direction: Hallucination direction vector for linear scoring.

    Spec: REQ-INFER-015
    """

    def __init__(
        self,
        critical_layers: list[int],
        model: GibbsModel,
        direction: jax.Array,
        threshold: float = 0.5,
        constraint_name: str = "layer_ebm",
    ) -> None:
        """Initialize the LayerEBMVerifier.

        Args:
            critical_layers: Layer indices to extract (from
                ``identify_critical_layers``).
            model: Trained ``GibbsModel`` whose input_dim matches the
                concatenated critical-layer dimension.
            direction: Hallucination direction vector, shape
                ``(total_dim,)`` or ``(k, total_dim)``.
            threshold: Energy threshold for the ``is_satisfied`` check.
            constraint_name: Human-readable name for reports.
        """
        self._critical_layers = critical_layers
        self._model = model
        self._direction = direction
        self._threshold = threshold
        self._name = constraint_name

    @property
    def critical_layers(self) -> list[int]:
        """Layer indices selected for monitoring."""
        return self._critical_layers

    @property
    def name(self) -> str:
        """Human-readable constraint name."""
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        """Energy threshold below which the constraint is satisfied."""
        return self._threshold

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute combined hallucination energy for concatenated activations.

        **Detailed explanation for engineers:**
            Takes the concatenated critical-layer activations (already
            pooled and concatenated by the caller or by
            ``energy_from_layers``) and computes:

                E_total = E_gibbs(x) + hallucination_energy(x, direction)

            The Gibbs energy captures nonlinear patterns learned from
            training data. The direction energy captures the linear
            separation axis. Their sum provides a robust hallucination
            score.

        Args:
            x: Concatenated critical-layer activations, shape
                ``(total_dim,)``.

        Returns:
            Scalar energy value.

        Spec: REQ-INFER-015
        """
        ebm_energy = self._model.energy(x)
        dir_energy = hallucination_energy(x, self._direction)
        return ebm_energy + dir_energy

    def energy_from_layers(
        self,
        activations: dict[int, jax.Array],
    ) -> jax.Array:
        """Compute energy directly from per-layer activation dict.

        **Detailed explanation for engineers:**
            Convenience method that handles the layer selection and
            concatenation internally. Pass in the full dict of per-layer
            activations (as returned by ``extract_layer_activations``),
            and this method extracts and concatenates the critical layers
            before computing the energy.

        Args:
            activations: Dict mapping layer index to activation array.
                Each value has shape ``(seq_len, hidden_dim)`` or
                ``(hidden_dim,)``.

        Returns:
            Scalar energy value.

        Spec: REQ-INFER-015
        """
        parts: list[jax.Array] = []
        for layer_idx in self._critical_layers:
            act = activations[layer_idx]
            if act.ndim == 2:
                act = jnp.mean(act, axis=0)
            parts.append(act)
        x = jnp.concatenate(parts)
        return self.energy(x)


def build_layer_ebm_verifier(
    activations_correct: dict[int, list[jax.Array]],
    activations_hallucinated: dict[int, list[jax.Array]],
    config: LayerEBMConfig | None = None,
) -> LayerEBMVerifier:
    """Build a complete LayerEBMVerifier from raw per-layer activations.

    **Researcher summary:**
        End-to-end factory: identify critical layers, train NCE model,
        find hallucination direction, return assembled verifier.

    **Detailed explanation for engineers:**
        This is the main entry point for constructing a layer-selective
        hallucination monitor. It orchestrates the full pipeline:

        1. Call ``identify_critical_layers`` to find the top-k layers.
        2. Concatenate activations from those layers for both classes.
        3. Find the hallucination direction on the concatenated space.
        4. Train a ``GibbsModel`` via NCE on the concatenated data.
        5. Wrap everything in a ``LayerEBMVerifier``.

        After calling this function, you have a ready-to-use verifier
        that can score new activations via ``verifier.energy(x)`` or
        ``verifier.energy_from_layers(activations_dict)``.

    Args:
        activations_correct: Per-layer activations from correct outputs.
        activations_hallucinated: Per-layer activations from hallucinated outputs.
        config: Training and layer selection configuration. Uses defaults
            if None.

    Returns:
        A fully assembled ``LayerEBMVerifier``.

    Spec: REQ-INFER-015
    """
    if config is None:
        config = LayerEBMConfig()
    config.validate()

    # Step 1: Identify critical layers.
    critical_layers = identify_critical_layers(
        activations_correct,
        activations_hallucinated,
        top_k=config.top_k_layers,
    )

    # Step 2: Concatenate critical-layer activations.
    correct_concat = _concat_critical_activations(activations_correct, critical_layers)
    halluc_concat = _concat_critical_activations(activations_hallucinated, critical_layers)

    # Step 3: Find hallucination direction on concatenated space.
    correct_list = [correct_concat[i] for i in range(correct_concat.shape[0])]
    halluc_list = [halluc_concat[i] for i in range(halluc_concat.shape[0])]
    direction = find_hallucination_direction(correct_list, halluc_list)

    # Step 4: Train Gibbs model via NCE.
    verifier_config = LearnedVerifierConfig(
        hidden_dims=config.hidden_dims,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        seed=config.seed,
    )
    model = train_layer_ebm(correct_concat, halluc_concat, verifier_config)

    # Step 5: Assemble verifier.
    return LayerEBMVerifier(
        critical_layers=critical_layers,
        model=model,
        direction=direction,
    )
