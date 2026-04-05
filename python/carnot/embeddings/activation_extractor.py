"""Layer-wise activation extraction for hallucination detection.

**Researcher summary:**
    Extracts hidden-state activations at every transformer layer during
    inference, then computes per-layer statistics (norm, direction change,
    attention entropy) that serve as real-time hallucination signals.

**Detailed explanation for engineers:**
    Large language models (LLMs) produce text by passing input tokens through
    a stack of transformer layers. Each layer outputs a hidden-state tensor
    of shape ``(batch, seq_len, hidden_dim)``. During normal generation the
    hidden states evolve smoothly from layer to layer — each layer refines
    the representation incrementally. When the model hallucinates (generates
    text not grounded in the input), distinctive patterns emerge:

    1. **Norm spikes** — the L2 norm of a layer's output jumps sharply,
       indicating the model is injecting energy into a new, ungrounded
       direction.
    2. **Direction change** — the cosine distance between consecutive layers
       increases, meaning the representation is veering away from the
       trajectory established by earlier layers.
    3. **Attention entropy** — the Shannon entropy of attention weight
       distributions changes. Uniform (high-entropy) attention suggests the
       model has no strong evidence for any particular token, which
       correlates with confabulation.

    This module captures these signals by hooking into a Hugging Face
    ``transformers`` model's forward pass using PyTorch's
    ``register_forward_hook`` API. The hook fires after each layer computes
    its output, letting us record the hidden state without modifying the
    model architecture.

    **Why JAX arrays?**
    Carnot's energy-based models and downstream analysis pipelines are
    JAX-native. Converting the captured PyTorch tensors to JAX arrays
    ensures seamless integration with the rest of the framework.

    **Graceful fallback:**
    The ``transformers`` and ``torch`` libraries are optional. If either is
    missing, extraction functions return ``None`` rather than raising.
    Callers should check for ``None`` and skip activation-based analysis
    when the dependencies are unavailable.

Spec: REQ-INFER-014
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp


@dataclass
class ActivationConfig:
    """Configuration for layer-wise activation extraction.

    **Detailed explanation for engineers:**
        Bundles the settings needed to run a transformer model and capture
        its internal activations:

        - ``model_name``: A Hugging Face model identifier. We default to a
          small model (Qwen3-0.6B) so that activation extraction is feasible
          on machines without large GPUs. Any causal-LM or encoder model
          that exposes ``model.layers`` (or equivalent) will work.

        - ``device``: Hardware for PyTorch inference. ``"cpu"`` is universal;
          ``"cuda"`` leverages an NVIDIA GPU for faster forward passes.

    Attributes:
        model_name: Hugging Face model identifier for the transformer.
        device: Hardware device for PyTorch inference (``"cpu"`` or ``"cuda"``).

    Spec: REQ-INFER-014
    """

    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "cpu"


def extract_layer_activations(
    text: str,
    config: Optional[ActivationConfig] = None,
) -> Optional[dict[int, jnp.ndarray]]:
    """Extract hidden-state activations from every transformer layer.

    **Researcher summary:**
        Runs ``text`` through the model, hooks each layer to capture its
        output hidden state, and returns a ``{layer_index: jax_array}``
        mapping.

    **Detailed explanation for engineers:**
        PyTorch's ``register_forward_hook`` lets us attach a callback to any
        ``nn.Module``. The callback receives ``(module, input, output)``
        after the module's ``forward()`` completes. We attach one hook per
        transformer layer so that when the model processes our input, each
        layer's output tensor is recorded.

        Steps:

        1. Lazy-import ``torch`` and ``transformers``. Return ``None`` if
           unavailable.
        2. Load the model and tokenizer from ``config.model_name``.
        3. Walk the model's children to find the list of transformer layers.
           Different model architectures nest their layers differently
           (e.g., ``model.layers``, ``transformer.h``, ``encoder.layer``).
           We try several common attribute paths.
        4. Register a forward hook on each layer that stores the layer's
           output hidden state (detached from the computation graph to save
           memory).
        5. Run the forward pass, collect activations, remove hooks.
        6. Convert each captured tensor from PyTorch to a JAX array.

    Args:
        text: Input text to feed through the model.
        config: Extraction configuration. Uses defaults if ``None``.

    Returns:
        A dict mapping layer index (0-based) to a JAX array of shape
        ``(seq_len, hidden_dim)`` containing that layer's output hidden
        state, or ``None`` if ``transformers``/``torch`` are not installed.

    Spec: REQ-INFER-014
    """
    if config is None:
        config = ActivationConfig()

    # Lazy import: torch and transformers are optional heavy dependencies.
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name)
    model = model.to(config.device)
    model.eval()

    # Tokenize input text.
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(config.device) for k, v in inputs.items()}

    # Find the list of transformer layers. Different architectures use
    # different attribute paths — we try several common ones.
    layers = _find_transformer_layers(model)
    if layers is None:
        return None

    # Register forward hooks to capture each layer's output hidden state.
    activations: dict[int, Any] = {}
    hooks = []

    for idx, layer in enumerate(layers):

        def _make_hook(layer_idx: int):
            """Create a closure that captures the layer index.

            We need this factory function because Python closures bind
            variables by reference, not by value. Without it, all hooks
            would share the same ``idx`` variable and record only the
            last layer's activation.
            """

            def hook_fn(module: Any, input: Any, output: Any) -> None:
                # Some layers return a tuple (hidden_state, attention_weights, ...);
                # others return the tensor directly. We always want the hidden state.
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                activations[layer_idx] = hidden.detach().cpu().numpy()

            return hook_fn

        hooks.append(layer.register_forward_hook(_make_hook(idx)))

    # Run the forward pass with hooks active.
    with torch.no_grad():
        model(**inputs)

    # Clean up hooks to avoid memory leaks.
    for h in hooks:
        h.remove()

    # Convert captured numpy arrays to JAX arrays. Squeeze batch dim.
    result: dict[int, jnp.ndarray] = {}
    for layer_idx, np_array in activations.items():
        # Shape is (1, seq_len, hidden_dim); squeeze to (seq_len, hidden_dim).
        squeezed = np_array.squeeze(0) if np_array.ndim == 3 else np_array
        result[layer_idx] = jnp.array(squeezed)

    return result


def _find_transformer_layers(model: Any) -> Optional[list[Any]]:
    """Locate the list of transformer layers in a model.

    **Detailed explanation for engineers:**
        Hugging Face models have inconsistent internal structures. A
        LLaMA/Qwen model puts layers at ``model.layers``, GPT-2 uses
        ``transformer.h``, BERT uses ``encoder.layer``. This helper tries
        each known path and returns the first match, or ``None`` if the
        architecture is unrecognized.

    Args:
        model: A Hugging Face ``PreTrainedModel`` instance.

    Returns:
        A list of ``nn.Module`` layer objects, or ``None`` if the layer
        list could not be found.

    Spec: REQ-INFER-014
    """
    # Common paths to the transformer block list, ordered by popularity.
    candidates = [
        # LLaMA, Qwen, Mistral, Gemma — decoder-only with model.layers
        ("model", "layers"),
        # GPT-2, GPT-Neo — transformer.h
        ("transformer", "h"),
        # BERT, RoBERTa, CodeBERT — encoder.layer
        ("encoder", "layer"),
        # DistilBERT — transformer.layer
        ("transformer", "layer"),
    ]

    for attr1, attr2 in candidates:
        parent = getattr(model, attr1, None)
        if parent is not None:
            layers = getattr(parent, attr2, None)
            if layers is not None:
                return list(layers)

    return None


def compute_activation_stats(
    activations: dict[int, jnp.ndarray],
) -> dict[int, dict[str, float]]:
    """Compute per-layer statistics from captured activations.

    **Researcher summary:**
        For each layer, computes the L2 norm, cosine direction change from
        the previous layer, and an entropy proxy of the activation
        distribution. These statistics form the feature vector for
        downstream hallucination classifiers.

    **Detailed explanation for engineers:**
        Given a dict of ``{layer_index: activation_array}``, this function
        computes three statistics per layer:

        1. **Norm** — the L2 (Euclidean) norm of the mean activation vector.
           This tells us how "energetic" the layer's output is. A sudden
           spike in norm between layers can indicate the model is injecting
           a new, ungrounded signal.

        2. **Direction change** — the cosine distance (1 - cosine_similarity)
           between the current layer's mean vector and the previous layer's
           mean vector. A value near 0 means the representation barely
           changed; a value near 1 (or even 2, for opposite directions)
           means a drastic shift. The first layer (index 0) has direction
           change of 0.0 by convention since there is no preceding layer.

        3. **Entropy** — the Shannon entropy of the softmax-normalized
           absolute activation values, averaged across the sequence. This
           is a proxy for how "spread out" the activation energy is across
           the hidden dimensions. High entropy means the energy is evenly
           distributed (uncertain); low entropy means it is concentrated in
           a few dimensions (confident).

    Args:
        activations: Dict mapping layer index to a JAX array of shape
            ``(seq_len, hidden_dim)``.

    Returns:
        Dict mapping layer index to a stats dict with keys ``"norm"``,
        ``"direction_change"``, and ``"entropy"``, each a Python float.

    Spec: REQ-INFER-014
    """
    stats: dict[int, dict[str, float]] = {}
    sorted_indices = sorted(activations.keys())

    prev_mean: Optional[jnp.ndarray] = None

    for layer_idx in sorted_indices:
        act = activations[layer_idx]

        # Mean activation vector across the sequence dimension.
        # Shape: (hidden_dim,)
        mean_vec = jnp.mean(act, axis=0)

        # 1. L2 norm of the mean activation vector.
        norm = float(jnp.linalg.norm(mean_vec))

        # 2. Cosine direction change from the previous layer.
        if prev_mean is not None:
            cos_sim = _cosine_similarity(prev_mean, mean_vec)
            direction_change = float(1.0 - cos_sim)
        else:
            direction_change = 0.0

        # 3. Entropy of the activation distribution.
        # We take the absolute values, normalize via softmax to get a
        # probability-like distribution, then compute Shannon entropy.
        entropy = float(_activation_entropy(act))

        stats[layer_idx] = {
            "norm": norm,
            "direction_change": direction_change,
            "entropy": entropy,
        }

        prev_mean = mean_vec

    return stats


def _cosine_similarity(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute cosine similarity between two vectors.

    **Detailed explanation for engineers:**
        Cosine similarity measures the angle between two vectors, ignoring
        their magnitudes. It equals ``dot(a, b) / (||a|| * ||b||)`` and
        ranges from -1 (opposite directions) to +1 (same direction).

        We clamp the denominator to avoid division by zero when a vector
        has zero norm (which would happen if all activations were exactly 0).

    Spec: REQ-INFER-014
    """
    dot = jnp.dot(a, b)
    norms = jnp.linalg.norm(a) * jnp.linalg.norm(b)
    # Clamp denominator to avoid division by zero.
    return dot / jnp.maximum(norms, 1e-8)


def _activation_entropy(act: jnp.ndarray) -> jnp.ndarray:
    """Compute the entropy of the activation distribution.

    **Detailed explanation for engineers:**
        To treat activation magnitudes as a probability distribution, we:

        1. Take absolute values (activations can be negative).
        2. Apply softmax across the hidden dimension to normalize each
           position's activation vector into a valid probability distribution
           that sums to 1.
        3. Compute Shannon entropy: ``H = -sum(p * log(p))``.
        4. Average the entropy across all sequence positions.

        The result is a scalar measuring how "spread out" the activation
        energy is. Maximum entropy occurs when all hidden dimensions have
        equal magnitude; minimum entropy occurs when all energy is
        concentrated in a single dimension.

    Spec: REQ-INFER-014
    """
    # Take absolute values and apply softmax to get pseudo-probabilities.
    abs_act = jnp.abs(act)
    # Softmax along hidden dimension (axis=-1).
    probs = jax_softmax(abs_act, axis=-1)
    # Shannon entropy: -sum(p * log(p)), with log(0) guarded by a small epsilon.
    log_probs = jnp.log(probs + 1e-12)
    entropy_per_position = -jnp.sum(probs * log_probs, axis=-1)
    # Average across sequence positions.
    return jnp.mean(entropy_per_position)


def jax_softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Numerically stable softmax.

    **Detailed explanation for engineers:**
        The naive formula ``exp(x) / sum(exp(x))`` overflows for large
        values. The standard trick is to subtract the maximum value before
        exponentiating: ``exp(x - max(x)) / sum(exp(x - max(x)))``. This
        produces identical results but keeps all intermediate values in a
        safe numerical range.

    Spec: REQ-INFER-014
    """
    x_max = jnp.max(x, axis=axis, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)
