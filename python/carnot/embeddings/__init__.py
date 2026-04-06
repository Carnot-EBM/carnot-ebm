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

from carnot.embeddings.activation_extractor import (
    ActivationConfig,
    compute_activation_stats,
    extract_layer_activations,
)
from carnot.embeddings.activation_steering import (
    SteeringConfig,
    calibrate_alpha,
    steered_generate,
)
from carnot.embeddings.concept_vectors import (
    CONCEPT_PROMPTS,
    best_concept_for_detection,
    concept_energy,
    find_concept_vectors,
)
from carnot.embeddings.hallucination_direction import (
    HallucinationDirectionConfig,
    HallucinationDirectionConstraint,
    find_hallucination_direction,
    hallucination_energy,
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
from carnot.embeddings.layer_ebm import (
    LayerEBMConfig,
    LayerEBMVerifier,
    build_layer_ebm_verifier,
    identify_critical_layers,
    train_layer_ebm,
)
from carnot.embeddings.layer_navigator import (
    LayerNavigatorConfig,
    find_best_layers,
    score_layer_steerability,
)
from carnot.embeddings.model_embeddings import (
    ModelEmbeddingConfig,
    extract_embedding,
)
from carnot.embeddings.weight_steering import (
    apply_cws,
    revert_cws,
    steered_model,
)

__all__ = [
    "ActivationConfig",
    "CONCEPT_PROMPTS",
    "ContextPredictionEnergy",
    "HallucinationDirectionConfig",
    "HallucinationDirectionConstraint",
    "JEPAEnergyConfig",
    "LayerEBMConfig",
    "LayerEBMVerifier",
    "LayerNavigatorConfig",
    "ModelEmbeddingConfig",
    "SteeringConfig",
    "apply_cws",
    "best_concept_for_detection",
    "build_layer_ebm_verifier",
    "calibrate_alpha",
    "compute_activation_stats",
    "concept_energy",
    "embedding_repair",
    "extract_embedding",
    "extract_layer_activations",
    "find_best_layers",
    "find_concept_vectors",
    "find_hallucination_direction",
    "generate_jepa_training_data",
    "hallucination_energy",
    "identify_critical_layers",
    "nce_loss",
    "nearest_code_match",
    "revert_cws",
    "score_layer_steerability",
    "steered_generate",
    "steered_model",
    "train_jepa_energy",
    "train_layer_ebm",
]
