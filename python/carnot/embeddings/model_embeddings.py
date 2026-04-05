"""Transformer-based semantic code embedding extraction.

**Researcher summary:**
    Extracts dense vector embeddings from source code using pretrained
    transformer models (CodeBERT, CodeT5, etc.) via the Hugging Face
    ``transformers`` library. Returns embeddings as JAX arrays.
    Gracefully degrades to ``None`` when ``transformers`` is unavailable.

**Detailed explanation for engineers:**
    A pretrained language model for code (like CodeBERT) is a neural network
    that has been trained on millions of code examples to understand programming
    language structure and semantics. When we feed a code snippet into the model,
    it produces a sequence of hidden state vectors — one per input token. To get
    a single fixed-size vector for the whole snippet, we either:

    1. Take the **[CLS] token** embedding (the special first token that
       transformers use as a summary representation), or
    2. Compute the **mean pool** — average all token embeddings together.

    This module uses mean pooling by default because it tends to produce more
    robust representations for code, where the [CLS] token alone may not capture
    enough information from longer snippets.

    **How tokenization works:**
    The model's tokenizer breaks the input code string into subword tokens
    (e.g., ``"def foo"`` might become ``["def", "fo", "##o"]``). Each token
    gets mapped to an integer ID. The model then processes these IDs through
    its transformer layers to produce contextualized embeddings.

    **Why JAX arrays?**
    The rest of Carnot uses JAX for accelerated numerical computation. By
    converting the PyTorch tensor output from ``transformers`` into a JAX array,
    the embeddings integrate seamlessly with Carnot's energy functions, training
    loops, and samplers without any framework mismatch.

    **Graceful fallback:**
    The ``transformers`` library is large (~500 MB with dependencies) and not
    needed for core EBM functionality. Rather than making it a hard dependency,
    we lazily import it and return ``None`` if it is missing. Callers should
    check for ``None`` and fall back to simpler embedding methods (e.g.,
    bag-of-tokens) when transformer embeddings are unavailable.

Spec: REQ-EMBED-001
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax.numpy as jnp


@dataclass
class ModelEmbeddingConfig:
    """Configuration for transformer-based code embedding extraction.

    **Detailed explanation for engineers:**
        This dataclass bundles together all the settings needed to run a
        pretrained transformer model for embedding extraction:

        - ``model_name``: The Hugging Face model identifier. CodeBERT is a
          BERT-based model pretrained on code from six programming languages.
          It produces 768-dimensional embeddings. Other options include
          ``"Salesforce/codet5-small"`` (also 768-dim) or any model that
          outputs ``last_hidden_state`` from its forward pass.

        - ``device``: Which hardware to run inference on. ``"cpu"`` works
          everywhere; ``"cuda"`` uses an NVIDIA GPU if available (much faster
          for large batches).

        - ``max_length``: Maximum number of tokens the model will process.
          Code longer than this gets truncated. 512 tokens covers most
          individual functions; increase for whole-file embeddings.

    Attributes:
        model_name: Hugging Face model identifier for the pretrained encoder.
        device: Hardware device for PyTorch inference (``"cpu"`` or ``"cuda"``).
        max_length: Maximum token sequence length before truncation.
    """

    model_name: str = "microsoft/codebert-base"
    device: str = "cpu"
    max_length: int = 512


def extract_embedding(
    code: str, config: Optional[ModelEmbeddingConfig] = None
) -> Optional[jnp.ndarray]:
    """Extract a semantic embedding vector from a code snippet.

    **Researcher summary:**
        Tokenizes ``code``, runs it through a pretrained transformer, and
        returns the mean-pooled last hidden state as a 1-D JAX array.

    **Detailed explanation for engineers:**
        This function performs the following steps:

        1. **Lazy import** ``transformers`` and ``torch``. If either is not
           installed, the function immediately returns ``None`` rather than
           crashing. This keeps the ``transformers`` dependency optional.

        2. **Tokenize** the input code string using the model's tokenizer.
           The tokenizer converts the code into a list of integer token IDs
           and an attention mask (1 for real tokens, 0 for padding). We set
           ``truncation=True`` so that inputs longer than ``max_length`` are
           cut to fit, and ``return_tensors="pt"`` to get PyTorch tensors.

        3. **Forward pass** through the model with ``torch.no_grad()`` to
           disable gradient tracking (we only need inference, not training).
           The model returns an object whose ``last_hidden_state`` is a
           3-D tensor of shape ``(batch=1, seq_len, hidden_dim)``.

        4. **Mean pooling**: We multiply each token's hidden state by its
           attention mask (to zero out padding tokens), sum across the
           sequence dimension, and divide by the number of non-padding
           tokens. This gives a single vector of shape ``(hidden_dim,)``
           that summarizes the entire code snippet.

        5. **Convert to JAX**: The NumPy array from PyTorch is wrapped in
           ``jnp.array()`` to produce a JAX array for downstream use.

    Args:
        code: Source code string to embed. Can be any programming language
            that the model was trained on.
        config: Embedding configuration. If ``None``, uses default settings
            (CodeBERT on CPU with 512 max tokens).

    Returns:
        A 1-D JAX array of shape ``(hidden_dim,)`` containing the semantic
        embedding, or ``None`` if the ``transformers`` library is not
        installed.

    Spec: REQ-EMBED-001
    """
    if config is None:
        config = ModelEmbeddingConfig()

    # Lazy import: transformers and torch are optional heavy dependencies.
    # If they are not installed, we return None so callers can fall back
    # to simpler embedding methods (e.g., bag-of-tokens TF-IDF).
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name)
    model = model.to(config.device)
    model.eval()

    # Tokenize the input code. padding=True is not needed for a single input
    # but included for clarity; truncation ensures we don't exceed max_length.
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length,
        padding=True,
    )
    inputs = {k: v.to(config.device) for k, v in inputs.items()}

    # Run the model in inference mode (no gradient computation needed).
    # last_hidden_state shape: (batch=1, sequence_length, hidden_dim)
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    attention_mask = inputs["attention_mask"]  # (1, seq_len)

    # Mean pooling: average the token embeddings, ignoring padding tokens.
    # We expand the attention mask to match the hidden state dimensions,
    # multiply to zero out padding, sum across the sequence axis, and
    # divide by the count of real (non-padding) tokens.
    mask_expanded = attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
    sum_hidden = (last_hidden * mask_expanded).sum(dim=1)  # (1, hidden_dim)
    count = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (1, hidden_dim)
    mean_pooled = sum_hidden / count  # (1, hidden_dim)

    # Convert from PyTorch tensor -> NumPy -> JAX array.
    embedding_np = mean_pooled.squeeze(0).cpu().numpy()
    return jnp.array(embedding_np)
