"""Load pre-trained EBM models from HuggingFace or local exports.

**Researcher summary:**
    Provides a simple API to load a trained Gibbs EBM for hallucination
    detection. Models can be loaded from local exports/ directory or
    downloaded from HuggingFace. The loaded EBM scores activation vectors:
    low energy = likely correct, high energy = likely hallucination.

**Detailed explanation for engineers:**
    The loading process:
    1. Find the model directory (local path or HuggingFace download).
    2. Read config.json for architecture (input_dim, hidden_dims, activation).
    3. Read model.safetensors for trained weights.
    4. Reconstruct a GibbsModel and set its parameters.
    5. Return the ready-to-use model.

    Three pre-trained models are available:
    - per-token-ebm-qwen3-06b (83.4% accuracy, for Qwen3-0.6B)
    - per-token-ebm-qwen35-08b-nothink (77.0%, for Qwen3.5-0.8B no thinking)
    - per-token-ebm-qwen35-08b-think (68.3%, for Qwen3.5-0.8B with thinking)

Spec: REQ-INFER-015
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import jax.numpy as jnp
import jax.random as jrandom

logger = logging.getLogger(__name__)

# Default HuggingFace organization
DEFAULT_ORG = "Carnot-EBM"

# Known model IDs and their source LLMs
KNOWN_MODELS: dict[str, dict[str, str]] = {
    "per-token-ebm-qwen3-06b": {
        "source_model": "Qwen/Qwen3-0.6B",
        "thinking": "N/A",
        "accuracy": "83.4%",
    },
    "per-token-ebm-qwen35-08b-nothink": {
        "source_model": "Qwen/Qwen3.5-0.8B",
        "thinking": "disabled",
        "accuracy": "77.0%",
    },
    "per-token-ebm-qwen35-08b-think": {
        "source_model": "Qwen/Qwen3.5-0.8B",
        "thinking": "enabled",
        "accuracy": "68.3%",
    },
    "per-token-ebm-lfm25-350m-nothink": {
        "source_model": "LiquidAI/LFM2.5-350M",
        "thinking": "N/A",
        "accuracy": "pending",
    },
    "per-token-ebm-lfm25-12b-nothink": {
        "source_model": "LiquidAI/LFM2.5-1.2B-Instruct",
        "thinking": "disabled",
        "accuracy": "pending",
    },
    "per-token-ebm-bonsai-17b-nothink": {
        "source_model": "prism-ml/Bonsai-1.7B-unpacked",
        "thinking": "disabled",
        "accuracy": "pending",
    },
    "per-token-ebm-qwen35-2b-nothink": {
        "source_model": "Qwen/Qwen3.5-2B",
        "thinking": "disabled",
        "accuracy": "pending",
    },
    "per-token-ebm-qwen35-4b-nothink": {
        "source_model": "Qwen/Qwen3.5-4B",
        "thinking": "disabled",
        "accuracy": "pending",
    },
    "per-token-ebm-qwen35-9b-nothink": {
        "source_model": "Qwen/Qwen3.5-9B",
        "thinking": "disabled",
        "accuracy": "pending",
    },
    "per-token-ebm-qwen35-27b-nothink": {
        "source_model": "Qwen/Qwen3.5-27B",
        "thinking": "disabled",
        "accuracy": "pending",
    },
    "per-token-ebm-qwen35-35b-nothink": {
        "source_model": "Qwen/Qwen3.5-35B-A3B",
        "thinking": "disabled",
        "accuracy": "pending",
    },
    "per-token-ebm-gemma4-e2b-nothink": {
        "source_model": "google/gemma-4-E2B",
        "thinking": "N/A",
        "accuracy": "pending",
    },
    "per-token-ebm-gemma4-e2b-it-nothink": {
        "source_model": "google/gemma-4-E2B-it",
        "thinking": "disabled",
        "accuracy": "pending",
    },
    "per-token-ebm-gemma4-e4b-nothink": {
        "source_model": "google/gemma-4-E4B",
        "thinking": "N/A",
        "accuracy": "pending",
    },
    "per-token-ebm-gemma4-e4b-it-nothink": {
        "source_model": "google/gemma-4-E4B-it",
        "thinking": "disabled",
        "accuracy": "pending",
    },
    "per-token-ebm-gptoss-20b-nothink": {
        "source_model": "openai/gpt-oss-20b",
        "thinking": "disabled",
        "accuracy": "pending",
    },
}


def load_ebm(
    model_id: str = "per-token-ebm-qwen35-08b-nothink",
    local_dir: str | None = None,
    hf_org: str = DEFAULT_ORG,
) -> Any:
    """Load a pre-trained Gibbs EBM for hallucination detection.

    **Detailed explanation for engineers:**
        Searches for the model in this order:
        1. ``local_dir`` if provided (exact path to model directory).
        2. ``exports/{model_id}/`` relative to the project root.
        3. HuggingFace Hub at ``{hf_org}/{model_id}``.

        The model directory must contain ``config.json`` and
        ``model.safetensors``.

    Args:
        model_id: Model identifier (e.g., "per-token-ebm-qwen35-08b-nothink").
        local_dir: Optional local directory containing the model files.
        hf_org: HuggingFace organization name (default "Carnot-EBM").

    Returns:
        A loaded GibbsModel ready to call .energy(activation_vector).

    Raises:
        FileNotFoundError: If model files cannot be found locally or on HF.

    Spec: REQ-INFER-015
    """
    from carnot.models.gibbs import GibbsConfig, GibbsModel

    model_dir = _find_model_dir(model_id, local_dir, hf_org)
    config_path = os.path.join(model_dir, "config.json")
    weights_path = os.path.join(model_dir, "model.safetensors")

    # Load config
    with open(config_path) as f:
        config_dict = json.load(f)

    config = GibbsConfig(
        input_dim=config_dict["input_dim"],
        hidden_dims=config_dict["hidden_dims"],
        activation=config_dict["activation"],
    )
    ebm = GibbsModel(config, key=jrandom.PRNGKey(0))

    # Load weights
    from safetensors.numpy import load_file

    weights = load_file(weights_path)

    layers = []
    for i in range(config_dict["n_layers"]):
        w = jnp.array(weights[f"layer_{i}_weight"])
        b = jnp.array(weights[f"layer_{i}_bias"])
        layers.append((w, b))
    ebm.layers = layers
    ebm.output_weight = jnp.array(weights["output_weight"])
    ebm.output_bias = jnp.array(weights["output_bias"])

    logger.info("Loaded EBM: %s from %s", model_id, model_dir)
    return ebm


def _find_model_dir(
    model_id: str,
    local_dir: str | None,
    hf_org: str,
) -> str:
    """Find the model directory, checking local then HuggingFace.

    Spec: REQ-INFER-015
    """
    # 1. Explicit local directory
    if local_dir and os.path.isdir(local_dir):  # noqa: SIM102
        if os.path.exists(os.path.join(local_dir, "config.json")):
            return local_dir

    # 2. Project exports/ directory
    project_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    exports_dir = os.path.join(project_root, "exports", model_id)
    if os.path.isdir(exports_dir) and os.path.exists(os.path.join(exports_dir, "config.json")):
        return exports_dir

    # 3. HuggingFace Hub
    try:
        from huggingface_hub import snapshot_download

        repo_id = f"{hf_org}/{model_id}"
        logger.info("Downloading %s from HuggingFace...", repo_id)
        return snapshot_download(repo_id)
    except Exception as e:
        raise FileNotFoundError(
            f"Model {model_id} not found locally (exports/{model_id}/) "
            f"or on HuggingFace ({hf_org}/{model_id}): {e}"
        ) from e


def get_model_info(model_id: str) -> dict[str, str]:
    """Get metadata about a known model.

    Returns:
        Dict with source_model, thinking mode, and accuracy.

    Spec: REQ-INFER-015
    """
    if model_id in KNOWN_MODELS:
        return KNOWN_MODELS[model_id]
    return {"source_model": "unknown", "thinking": "unknown", "accuracy": "unknown"}
