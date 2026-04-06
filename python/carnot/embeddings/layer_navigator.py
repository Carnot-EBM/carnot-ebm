"""Layer steerability navigation for activation steering.

**Researcher summary:**
    Scores each transformer layer by how much a directional perturbation
    at that layer changes the final logits. Layers with high steerability
    are the best targets for activation steering interventions.

**Detailed explanation for engineers:**
    Not all transformer layers respond equally to perturbations. When we
    inject a steering vector (e.g., the hallucination direction) into a
    layer's hidden state, the effect on the model's output logits varies
    dramatically depending on which layer we perturb. Early layers may
    have their signal washed out by subsequent processing; late layers
    may not have enough remaining computation to propagate the change.

    This module systematically scores every layer's "steerability" by:

    1. Running a clean forward pass to get baseline logits.
    2. For each layer, hooking into the forward pass to add
       ``alpha * direction`` to the hidden state, then measuring the
       L2 distance between the perturbed and baseline logits.
    3. Ranking layers by this logit displacement and returning the
       top-N most steerable layers.

    The result tells you *where* to steer. The ``activation_steering``
    and ``weight_steering`` modules then handle *how* to steer.

    **Why L2 distance on logits?**
    Logits are the model's pre-softmax output distribution. Large L2
    changes in logits mean the model's token predictions shift
    significantly, which is exactly what we want from a steering
    intervention. KL divergence would also work but is more expensive
    and less numerically stable.

    **Graceful fallback:**
    Like other embedding modules, this lazy-imports ``torch`` and
    ``transformers``. If unavailable, functions return ``None`` so
    callers can degrade gracefully.

Spec: REQ-INFER-015
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax.numpy as jnp


@dataclass
class LayerNavigatorConfig:
    """Configuration for layer steerability scoring.

    **Detailed explanation for engineers:**
        Controls the perturbation experiment used to rank layers:

        - ``alpha``: Scale of the perturbation injected at each layer.
          Larger alpha means a stronger push, which makes steerability
          differences more visible but may push the model into a
          nonlinear regime where rankings are unreliable. 1.0 is a
          reasonable default for normalized direction vectors.

        - ``n_best``: How many top-scoring layers to return from
          ``find_best_layers``. 3 is a good default: it gives enough
          layers for multi-layer steering while keeping inference
          overhead low.

    Attributes:
        alpha: Perturbation scale factor. Default 1.0.
        n_best: Number of top steerable layers to return. Default 3.

    Spec: REQ-INFER-015
    """

    alpha: float = 1.0
    n_best: int = 3


def score_layer_steerability(
    model: Any,
    tokenizer: Any,
    qa_pairs: list[tuple[str, str]],
    layer_idx: int,
    direction: jnp.ndarray,
    config: LayerNavigatorConfig | None = None,
) -> float | None:
    """Score how much perturbing a single layer changes the model's output logits.

    **Researcher summary:**
        Hooks into layer ``layer_idx``, adds ``alpha * direction`` to its
        hidden state, and measures L2 distance between original and
        perturbed logits averaged over the QA pairs.

    **Detailed explanation for engineers:**
        The algorithm works as follows for each (question, answer) pair:

        1. Tokenize the question.
        2. Run a clean forward pass to get baseline logits.
        3. Register a forward hook on the specified layer that modifies
           the hidden state: ``h = h + alpha * direction``. The direction
           is broadcast across the sequence dimension so every token
           position gets the same perturbation.
        4. Run the forward pass again with the hook active.
        5. Compute the L2 norm of (perturbed_logits - baseline_logits).
        6. Remove the hook.
        7. Average the L2 distances across all QA pairs.

        Higher average distance means the layer is more "steerable" --
        perturbations there have a bigger impact on what the model outputs.

    Args:
        model: A Hugging Face ``PreTrainedModel`` (e.g., causal LM).
        tokenizer: The corresponding tokenizer.
        qa_pairs: List of (question, expected_answer) tuples. The answer
            is not used for scoring but maintains API consistency with
            ``calibrate_alpha``.
        layer_idx: Which layer to perturb (0-indexed).
        direction: Perturbation direction vector, shape ``(hidden_dim,)``.
        config: Steerability scoring configuration. Uses defaults if None.

    Returns:
        Mean L2 logit displacement (float), or None if torch/transformers
        are not installed.

    Spec: REQ-INFER-015
    """
    if config is None:
        config = LayerNavigatorConfig()

    try:
        import torch
    except ImportError:  # pragma: no cover
        return None

    from carnot.embeddings.activation_extractor import _find_transformer_layers

    layers = _find_transformer_layers(model)
    if layers is None or layer_idx >= len(layers):
        return None

    import numpy as np

    direction_np = np.array(direction)
    total_distance = 0.0

    for question, _answer in qa_pairs:
        inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
        device = next(model.parameters(), None)
        dev = device.device if device is not None else "cpu"
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        # Baseline forward pass (no perturbation).
        with torch.no_grad():
            baseline_out = model(**inputs)
        baseline_logits = baseline_out.logits.detach().cpu().numpy()

        # Perturbed forward pass: hook adds alpha * direction to hidden state.
        def _make_hook(alpha: float, dir_vec: Any) -> Any:
            """Create a hook that adds a directional perturbation to the hidden state.

            We need a factory function because Python closures capture
            variables by reference. Without it, all hooks would share
            the same alpha/dir_vec binding.
            """

            def hook_fn(module: Any, input_: Any, output: Any) -> Any:
                if isinstance(output, tuple):
                    hidden = output[0]
                    # Add perturbation broadcast across sequence dimension.
                    perturbation = torch.tensor(
                        dir_vec * alpha, dtype=hidden.dtype, device=hidden.device
                    )
                    perturbed = hidden + perturbation.unsqueeze(0).unsqueeze(0)
                    return (perturbed,) + output[1:]
                else:  # pragma: no cover — only reachable with non-tuple layer output
                    perturbation = torch.tensor(
                        dir_vec * alpha, dtype=output.dtype, device=output.device
                    )
                    return output + perturbation.unsqueeze(0).unsqueeze(0)

            return hook_fn

        hook_handle = layers[layer_idx].register_forward_hook(
            _make_hook(config.alpha, direction_np)
        )

        with torch.no_grad():
            perturbed_out = model(**inputs)
        perturbed_logits = perturbed_out.logits.detach().cpu().numpy()

        hook_handle.remove()

        # L2 distance between baseline and perturbed logits.
        distance = float(np.linalg.norm(perturbed_logits - baseline_logits))
        total_distance += distance

    return total_distance / len(qa_pairs) if qa_pairs else 0.0


def find_best_layers(
    model: Any,
    tokenizer: Any,
    qa_pairs: list[tuple[str, str]],
    direction: jnp.ndarray,
    config: LayerNavigatorConfig | None = None,
) -> list[int] | None:
    """Score all layers and return the top-N most steerable layer indices.

    **Researcher summary:**
        Iterates over every transformer layer, scores its steerability
        via ``score_layer_steerability``, and returns the top-N layer
        indices sorted by descending steerability.

    **Detailed explanation for engineers:**
        This is a convenience wrapper that calls ``score_layer_steerability``
        for each layer in the model, collects the scores, and returns the
        top-N layer indices. The number of layers is discovered automatically
        by inspecting the model's architecture (same logic as
        ``_find_transformer_layers``).

        This function is the main entry point for layer navigation. Typical
        usage:

        1. Call ``find_best_layers`` to discover which layers to steer.
        2. Pass the result to ``SteeringConfig.layer_indices``.
        3. Use ``steered_generate`` or ``apply_cws`` for the actual steering.

    Args:
        model: A Hugging Face PreTrainedModel.
        tokenizer: The corresponding tokenizer.
        qa_pairs: List of (question, expected_answer) tuples for scoring.
        direction: Perturbation direction vector, shape ``(hidden_dim,)``.
        config: Configuration controlling alpha and n_best. Uses defaults
            if None.

    Returns:
        List of layer indices (ints) sorted by descending steerability,
        length ``min(n_best, n_layers)``. Returns None if torch/transformers
        are not available.

    Spec: REQ-INFER-015
    """
    if config is None:
        config = LayerNavigatorConfig()

    from carnot.embeddings.activation_extractor import _find_transformer_layers

    layers = _find_transformer_layers(model)
    if layers is None:
        return None

    scores: dict[int, float] = {}
    for idx in range(len(layers)):
        score = score_layer_steerability(model, tokenizer, qa_pairs, idx, direction, config)
        if score is not None:
            scores[idx] = score

    # Sort by score descending, return top n_best.
    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
    return ranked[: config.n_best]
