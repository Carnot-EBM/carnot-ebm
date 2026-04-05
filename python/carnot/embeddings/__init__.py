"""Semantic code embeddings for Carnot.

**Researcher summary:**
    Provides transformer-based dense vector embeddings of source code,
    capturing semantic meaning rather than surface-level token statistics.

**Detailed explanation for engineers:**
    Traditional code analysis often relies on bag-of-tokens or TF-IDF style
    representations, which treat code as flat text and miss structural and
    semantic relationships. For example, two functions that accomplish the same
    task with different variable names would appear dissimilar under token-based
    methods.

    This module wraps pretrained transformer models (e.g., CodeBERT, CodeT5)
    to produce dense vector embeddings (typically 768 dimensions) that capture
    the *meaning* of code. Similar code snippets will have embeddings that are
    close together in vector space (high cosine similarity), even if the surface
    tokens differ.

    The embeddings are returned as JAX arrays so they integrate naturally with
    the rest of the Carnot framework's JAX-based computation pipeline.

    The ``transformers`` library (Hugging Face) is an optional dependency.
    If it is not installed, embedding extraction gracefully returns ``None``
    so that downstream code can fall back to simpler representations.

Spec: REQ-EMBED-001
"""

from carnot.embeddings.hallucination_direction import (
    HallucinationDirectionConfig,
    HallucinationDirectionConstraint,
    find_hallucination_direction,
    hallucination_energy,
)
from carnot.embeddings.activation_extractor import (
    ActivationConfig,
    compute_activation_stats,
    extract_layer_activations,
)
from carnot.embeddings.jepa_energy import (
    ContextPredictionEnergy,
    JEPAEnergyConfig,
    embedding_repair,
    generate_jepa_training_data,
    nce_loss,
    nearest_code_match,
    train_jepa_energy,
)
from carnot.embeddings.model_embeddings import (
    ModelEmbeddingConfig,
    extract_embedding,
)

__all__ = [
    "ActivationConfig",
    "ContextPredictionEnergy",
    "HallucinationDirectionConfig",
    "HallucinationDirectionConstraint",
    "JEPAEnergyConfig",
    "ModelEmbeddingConfig",
    "compute_activation_stats",
    "embedding_repair",
    "extract_embedding",
    "extract_layer_activations",
    "find_hallucination_direction",
    "generate_jepa_training_data",
    "hallucination_energy",
    "nce_loss",
    "nearest_code_match",
    "train_jepa_energy",
]
