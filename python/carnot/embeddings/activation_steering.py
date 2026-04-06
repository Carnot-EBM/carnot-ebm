"""Activation steering via forward hooks during generation.

**Researcher summary:**
    Steers a language model away from hallucination by subtracting the
    hallucination direction vector from hidden states at critical layers
    during generation. Includes alpha calibration to find optimal
    steering strength.

**Detailed explanation for engineers:**
    Activation steering (also called "representation engineering" or
    "inference-time intervention") modifies a model's internal
    representations while it generates text, without changing any
    weights. The idea is simple:

    1. You have a "hallucination direction" vector that points from
       correct-output activations toward hallucinated-output activations
       in the model's hidden-state space.
    2. During generation, you hook into specific transformer layers and
       *subtract* a scaled version of this direction from the hidden
       state: ``h_new = h_old - alpha * direction``.
    3. This pushes the model's internal representation away from the
       hallucination region, making it less likely to generate incorrect
       text.

    **Why subtract rather than add?**
    The direction vector points *toward* hallucination (it's computed as
    ``mean_hallucinated - mean_correct``). Subtracting it pushes the
    representation back toward the correct region.

    **What is alpha?**
    Alpha controls the steering strength. Too small and the intervention
    has no effect; too large and it corrupts the model's representations
    entirely, producing gibberish. The ``calibrate_alpha`` function finds
    the sweet spot by trying several alpha values on a validation set
    and returning the one that maximizes accuracy (fraction of QA pairs
    where the model's generation contains the expected answer).

    **Graceful fallback:**
    Like other embedding modules, this lazy-imports ``torch`` and
    ``transformers``. If unavailable, functions return ``None``.

Spec: REQ-INFER-015
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax.numpy as jnp


@dataclass
class SteeringConfig:
    """Configuration for activation steering during generation.

    **Detailed explanation for engineers:**
        - ``layer_indices``: Which transformer layers to steer. These
          should come from ``find_best_layers`` (the most steerable
          layers). Steering all layers is wasteful and often harmful --
          you only need 2-3 critical layers.

        - ``alpha``: Steering strength. The hidden state at each
          steered layer is modified as ``h = h - alpha * direction``.
          Typical good values range from 0.5 to 2.0 depending on the
          model and direction magnitude (assuming normalized direction).

        - ``max_new_tokens``: Maximum number of tokens to generate.
          Passed directly to ``model.generate()``.

    Attributes:
        layer_indices: Which layers to apply steering hooks to.
        alpha: Steering strength multiplier. Default 1.0.
        max_new_tokens: Maximum tokens to generate. Default 50.

    Spec: REQ-INFER-015
    """

    layer_indices: list[int] = field(default_factory=list)
    alpha: float = 1.0
    max_new_tokens: int = 50


def steered_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    direction: jnp.ndarray,
    config: SteeringConfig | None = None,
) -> str | None:
    """Generate text with activation steering hooks that subtract the hallucination direction.

    **Researcher summary:**
        Registers forward hooks on critical layers that subtract
        ``alpha * direction`` from hidden states, runs ``model.generate()``,
        removes hooks, and returns the generated text.

    **Detailed explanation for engineers:**
        The generation pipeline works as follows:

        1. Tokenize the prompt.
        2. For each layer index in ``config.layer_indices``, register a
           forward hook that modifies the hidden state:
           ``hidden_state = hidden_state - alpha * direction``
           The direction is broadcast across batch and sequence dimensions.
        3. Call ``model.generate()`` with the tokenized input and
           ``max_new_tokens``. The hooks fire at every forward pass
           during autoregressive generation (once per new token).
        4. Remove all hooks to prevent memory leaks.
        5. Decode the generated token IDs back to text.

        **Hook removal safety:**
        We use a try/finally pattern to ensure hooks are always removed,
        even if generation raises an exception.

    Args:
        model: A Hugging Face causal language model.
        tokenizer: The corresponding tokenizer.
        prompt: Input text to continue generating from.
        direction: Hallucination direction vector, shape ``(hidden_dim,)``.
        config: Steering configuration. Uses defaults if None.

    Returns:
        Generated text (string), or None if torch/transformers unavailable.

    Spec: REQ-INFER-015
    """
    if config is None:
        config = SteeringConfig()

    try:
        import torch
    except ImportError:  # pragma: no cover
        return None

    import numpy as np

    from carnot.embeddings.activation_extractor import _find_transformer_layers

    layers = _find_transformer_layers(model)
    if layers is None:
        return None

    direction_np = np.array(direction)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    param = next(model.parameters(), None)
    dev = param.device if param is not None else "cpu"
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    # Register steering hooks on the specified layers.
    hooks = []

    def _make_steering_hook(alpha: float, dir_vec: Any) -> Any:
        """Create a hook that subtracts alpha * direction from hidden state.

        Factory function needed to avoid closure-over-loop-variable bugs.
        """

        def hook_fn(  # pragma: no cover — only fires during real torch forward
            module: Any, input_: Any, output: Any
        ) -> Any:
            if isinstance(output, tuple):
                hidden = output[0]
                perturbation = torch.tensor(
                    dir_vec * alpha, dtype=hidden.dtype, device=hidden.device
                )
                steered = hidden - perturbation.unsqueeze(0).unsqueeze(0)
                return (steered,) + output[1:]
            else:
                perturbation = torch.tensor(
                    dir_vec * alpha, dtype=output.dtype, device=output.device
                )
                return output - perturbation.unsqueeze(0).unsqueeze(0)

        return hook_fn

    try:
        for layer_idx in config.layer_indices:
            if layer_idx < len(layers):
                handle = layers[layer_idx].register_forward_hook(
                    _make_steering_hook(config.alpha, direction_np)
                )
                hooks.append(handle)

        # Generate with hooks active.
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
            )
    finally:
        # Always remove hooks to prevent memory leaks.
        for h in hooks:
            h.remove()

    # Decode generated tokens to text.
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def calibrate_alpha(
    model: Any,
    tokenizer: Any,
    qa_pairs: list[tuple[str, str]],
    layers: list[int],
    direction: jnp.ndarray,
    alphas: list[float] | None = None,
) -> float | None:
    """Find the steering alpha that maximizes accuracy on a validation set.

    **Researcher summary:**
        Tries each alpha value, generates answers for all QA pairs with
        that alpha's steering, computes accuracy (fraction containing
        expected answer), and returns the best alpha.

    **Detailed explanation for engineers:**
        Calibration works by brute-force grid search over candidate alpha
        values:

        1. For each candidate alpha in ``alphas``:
           a. Create a ``SteeringConfig`` with the given layers and alpha.
           b. For each (question, expected_answer) pair, call
              ``steered_generate`` and check if the expected answer
              appears in the generated text (case-insensitive substring
              match).
           c. Compute accuracy = fraction of QA pairs where the answer
              was found in the generation.
        2. Return the alpha with the highest accuracy. On ties, return
           the smallest alpha (prefer minimal intervention).

        **Why case-insensitive substring match?**
        This is a simple but robust accuracy metric. The model may
        generate the answer with different capitalization or surrounded
        by explanation text. Exact match would be too strict.

        **Default alpha grid:**
        [0.1, 0.5, 1.0, 2.0, 5.0] covers a wide range from gentle
        nudging to aggressive steering. For fine-grained calibration,
        pass a denser grid.

    Args:
        model: A Hugging Face causal language model.
        tokenizer: The corresponding tokenizer.
        qa_pairs: List of (question, expected_answer) tuples.
        layers: Layer indices to steer.
        direction: Hallucination direction vector.
        alphas: List of alpha values to try. Defaults to
            [0.1, 0.5, 1.0, 2.0, 5.0].

    Returns:
        The alpha value (float) that maximized accuracy, or None if
        torch/transformers unavailable.

    Spec: REQ-INFER-015
    """
    if alphas is None:
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0]

    try:
        import torch  # noqa: F401
    except ImportError:  # pragma: no cover
        return None

    best_alpha = alphas[0]
    best_accuracy = -1.0

    for alpha in alphas:
        config = SteeringConfig(
            layer_indices=layers,
            alpha=alpha,
            max_new_tokens=50,
        )

        correct_count = 0
        for question, expected_answer in qa_pairs:
            generated = steered_generate(model, tokenizer, question, direction, config)
            if generated is not None and expected_answer.lower() in generated.lower():
                correct_count += 1

        accuracy = correct_count / len(qa_pairs) if qa_pairs else 0.0

        # Prefer smallest alpha on ties (minimal intervention principle).
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_alpha = alpha

    return best_alpha
