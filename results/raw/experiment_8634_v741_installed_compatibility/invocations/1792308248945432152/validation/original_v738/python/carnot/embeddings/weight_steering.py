"""Contrastive Weight Steering (CWS) for permanent model modification.

**Researcher summary:**
    Modifies transformer layer weights in-place to steer the model away
    from hallucination without requiring runtime hooks. The weight
    modification is equivalent to projecting out the hallucination
    direction from the layer's output projection.

**Detailed explanation for engineers:**
    Activation steering (in ``activation_steering.py``) works by hooking
    into the forward pass at inference time. This is flexible but adds
    overhead: hooks fire on every forward pass and slow down generation.

    Contrastive Weight Steering (CWS) achieves the same effect by
    modifying the model's weights directly:

        W_new = W_old - alpha * outer(d, d) / ||d||^2

    where ``d`` is the hallucination direction vector and ``W_old`` is
    the output projection weight matrix of the target layer.

    **Why does this work?**
    The outer product ``d @ d^T / ||d||^2`` is a projection matrix that
    maps any vector to its component along ``d``. Subtracting this from
    the weight matrix means that the layer's output can no longer have a
    component along the hallucination direction -- it's been
    "projected out" of the weight space. This is mathematically
    equivalent to subtracting the projection from every hidden state
    that passes through the layer.

    **Trade-offs vs. activation steering:**
    - **Pro**: No runtime overhead. The model generates at full speed
      because there are no hooks.
    - **Pro**: Works with any inference framework (vLLM, TensorRT, etc.)
      since the weights are just regular model parameters.
    - **Con**: Irreversible (unless you save and restore original weights).
      The ``steered_model`` context manager handles save/restore
      automatically.
    - **Con**: Less flexible. Alpha is baked into the weights. To try a
      different alpha, you need to revert and re-apply.

    **Graceful fallback:**
    Like other embedding modules, this lazy-imports ``torch``. If
    unavailable, functions return ``None``.

Spec: REQ-INFER-015
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator

    import jax.numpy as jnp


def apply_cws(
    model: Any,
    layer_idx: int,
    direction: jnp.ndarray,
    alpha: float = 1.0,
) -> dict[str, Any] | None:
    """Modify a layer's output projection weights to steer away from hallucination direction.

    **Researcher summary:**
        Applies the CWS formula: ``W_new = W_old - alpha * outer(d, d) / ||d||^2``
        to the layer's output projection weight matrix. Returns the
        original weights so they can be reverted later.

    **Detailed explanation for engineers:**
        The modification targets the layer's output projection (the linear
        transformation at the end of the transformer block that projects
        the hidden state back to the model dimension). Different
        architectures name this differently:

        - LLaMA/Qwen: ``layer.self_attn.o_proj.weight``
        - GPT-2: ``layer.attn.c_proj.weight``
        - BERT: ``layer.output.dense.weight``

        We try several common paths and modify the first one found.

        The formula ``W -= alpha * outer(d, d) / ||d||^2`` is a rank-1
        update that removes the hallucination direction from the weight
        matrix. This is equivalent to applying a projection
        ``(I - alpha * d d^T / ||d||^2)`` to every output of the layer.

    Args:
        model: A Hugging Face PreTrainedModel.
        layer_idx: Which layer to modify (0-indexed).
        direction: Hallucination direction vector, shape ``(hidden_dim,)``.
        alpha: Steering strength. Default 1.0.

    Returns:
        Dict containing the original weight tensor (for reverting),
        or None if torch is unavailable or the layer structure is
        unrecognized.

    Spec: REQ-INFER-015
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None

    from carnot.embeddings.activation_extractor import _find_transformer_layers

    layers = _find_transformer_layers(model)
    if layers is None or layer_idx >= len(layers):
        return None

    layer = layers[layer_idx]

    # Find the output projection weight matrix.
    weight_param = _find_output_projection(layer)
    if weight_param is None:
        return None

    # Save original weights for reverting.
    original_data = weight_param.data.clone()

    # Compute rank-1 CWS update: W_new = W_old - alpha * outer(d, d) / ||d||^2
    d = np.array(direction).astype(np.float32)
    d_norm_sq = float(np.dot(d, d))
    if d_norm_sq < 1e-12:
        # Direction is effectively zero; no modification needed.
        return {"layer_idx": layer_idx, "original_weight": original_data}

    # outer(d, d) / ||d||^2 has shape (hidden_dim, hidden_dim).
    projection = np.outer(d, d) / d_norm_sq
    projection_tensor = torch.tensor(
        projection * alpha, dtype=weight_param.dtype, device=weight_param.device
    )

    # Apply the modification. Weight shape may be (out_dim, in_dim) or
    # (in_dim, out_dim) depending on the layer. We handle the common case
    # where weight shape is (hidden_dim, hidden_dim).
    with torch.no_grad():
        weight_param.data -= projection_tensor

    return {"layer_idx": layer_idx, "original_weight": original_data}


def revert_cws(
    model: Any,
    layer_idx: int,
    original_weights: dict[str, Any],
) -> bool | None:
    """Restore original weights after CWS modification.

    **Researcher summary:**
        Replaces the modified weight tensor with the saved original,
        undoing the CWS modification.

    **Detailed explanation for engineers:**
        Takes the ``original_weight`` tensor saved by ``apply_cws`` and
        writes it back into the layer's output projection. This fully
        reverts the CWS modification, restoring the model to its
        pre-steering state.

    Args:
        model: The same model that was modified by ``apply_cws``.
        layer_idx: Which layer to revert (must match the layer that
            was modified).
        original_weights: The dict returned by ``apply_cws``, containing
            the original weight tensor.

    Returns:
        True if revert succeeded, None if torch unavailable, False if
        the layer structure is unrecognized.

    Spec: REQ-INFER-015
    """
    try:
        import torch  # noqa: F401
    except ImportError:  # pragma: no cover
        return None

    from carnot.embeddings.activation_extractor import _find_transformer_layers

    layers = _find_transformer_layers(model)
    if layers is None or layer_idx >= len(layers):
        return False

    layer = layers[layer_idx]
    weight_param = _find_output_projection(layer)
    if weight_param is None:
        return False

    with torch.no_grad():
        weight_param.data.copy_(original_weights["original_weight"])

    return True


@contextmanager
def steered_model(
    model: Any,
    layers: list[int],
    direction: jnp.ndarray,
    alpha: float = 1.0,
) -> Generator[Any, None, None]:
    """Context manager that applies CWS on entry and reverts on exit.

    **Researcher summary:**
        Temporarily modifies model weights for the duration of a
        ``with`` block, then restores them automatically.

    **Detailed explanation for engineers:**
        This context manager provides a safe, exception-proof way to
        use CWS for temporary steering:

        .. code-block:: python

            with steered_model(model, [10, 15, 20], direction, alpha=1.5):
                output = model.generate(...)
            # Weights are back to normal here, even if generate() raised.

        On entry: calls ``apply_cws`` for each layer, saving originals.
        On exit (including exceptions): calls ``revert_cws`` for each
        layer, restoring the original weights.

    Args:
        model: A Hugging Face PreTrainedModel.
        layers: Layer indices to modify.
        direction: Hallucination direction vector.
        alpha: Steering strength.

    Yields:
        The model (same object, with modified weights).

    Spec: REQ-INFER-015
    """
    saved: list[tuple[int, dict[str, Any]]] = []
    try:
        for layer_idx in layers:
            result = apply_cws(model, layer_idx, direction, alpha)
            if result is not None:
                saved.append((layer_idx, result))
        yield model
    finally:
        # Revert all modifications in reverse order.
        for layer_idx, original in reversed(saved):
            revert_cws(model, layer_idx, original)


def _find_output_projection(layer: Any) -> Any:
    """Locate the output projection weight parameter in a transformer layer.

    **Detailed explanation for engineers:**
        Different transformer architectures place the output projection
        (the linear layer at the end of the attention block) in different
        places:

        - LLaMA/Qwen/Mistral: ``self_attn.o_proj.weight``
        - GPT-2/GPT-Neo: ``attn.c_proj.weight``
        - BERT/RoBERTa: ``output.dense.weight``
        - DistilBERT: ``sa_layer_norm`` (no separate proj — skip)

        We try each path and return the first ``weight`` parameter
        found, or None if the architecture is unrecognized.

    Args:
        layer: A single transformer layer module.

    Returns:
        The weight ``nn.Parameter`` of the output projection, or None.

    Spec: REQ-INFER-015
    """
    # Try common output projection paths.
    candidates = [
        # LLaMA, Qwen, Mistral — self_attn.o_proj
        ("self_attn", "o_proj"),
        # GPT-2, GPT-Neo — attn.c_proj
        ("attn", "c_proj"),
        # BERT, RoBERTa — output.dense
        ("output", "dense"),
    ]

    for attr1, attr2 in candidates:
        parent = getattr(layer, attr1, None)
        if parent is not None:
            proj = getattr(parent, attr2, None)
            if proj is not None:
                weight = getattr(proj, "weight", None)
                if weight is not None:
                    return weight

    return None
