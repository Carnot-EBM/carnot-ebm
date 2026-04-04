"""Energy-calibrated chain-of-thought: verify reasoning, not just answers.

**Researcher summary:**
    Instead of checking if the LLM's final answer is correct, this module
    checks if the LLM's REASONING is consistent. An energy model trained on
    (coherent_chain, incoherent_chain) pairs assigns low energy to good
    reasoning and high energy to bad reasoning. Langevin dynamics refines
    reasoning embeddings toward low-energy (coherent) regions.

**Detailed explanation for engineers:**
    The key insight from EBM-CoT (arxiv 2511.07124): catching bad reasoning
    EARLY prevents hallucinations from propagating through a chain of thought.
    Rather than waiting for the final answer to be wrong, detect inconsistency
    in the reasoning steps themselves.

    The pipeline:
    1. Convert reasoning text to an embedding vector (bag of tokens)
    2. Score it with a trained Gibbs model (low energy = coherent)
    3. Optionally refine via Langevin dynamics toward lower energy
    4. Classify as coherent or incoherent based on energy threshold

    Training data is generated via template mutations: coherent chains
    like "2+3=5, 5*2=10" vs incoherent chains like "2+3=7, 5*2=3".

Spec: REQ-INFER-011, SCENARIO-INFER-012
"""

from __future__ import annotations

import hashlib
import io
import logging
import random
import tokenize
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.training.nce import nce_loss

logger = logging.getLogger(__name__)


@dataclass
class ReasoningEnergyResult:
    """Result of reasoning chain energy analysis.

    Spec: REQ-INFER-011
    """

    chain_energy: float = 0.0
    is_coherent: bool = True
    threshold: float = 1.0


@dataclass
class ReasoningVerifierConfig:
    """Config for reasoning energy model training.

    Spec: REQ-INFER-011
    """

    vocab_size: int = 256
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    n_epochs: int = 50
    learning_rate: float = 0.01
    n_training_samples: int = 100
    seed: int = 42


def text_to_reasoning_embedding(text: str, vocab_size: int = 256) -> jax.Array:
    """Convert reasoning text to bag-of-tokens frequency vector.

    **Researcher summary:**
        Tokenizes text, hashes tokens to vocab buckets, returns frequency vector.
        Same approach as code_to_embedding in python_types.py.

    Spec: REQ-INFER-011
    """
    counts = [0.0] * vocab_size
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        for tok in tokens:
            if tok.string.strip():
                idx = int(hashlib.md5(tok.string.encode()).hexdigest(), 16) % vocab_size
                counts[idx] += 1.0
    except tokenize.TokenError:
        pass

    total = sum(counts)
    if total > 0:
        counts = [c / total for c in counts]

    return jnp.array(counts)


def compute_reasoning_energy(
    model: GibbsModel,
    chain_embedding: jax.Array,
) -> float:
    """Compute energy of a reasoning chain. Low = coherent, high = inconsistent.

    Spec: REQ-INFER-011
    """
    return float(model.energy(chain_embedding))


def refine_reasoning(
    chain_embedding: jax.Array,
    model: GibbsModel,
    n_langevin_steps: int = 3,
    step_size: float = 0.01,
    noise_scale: float = 0.001,
    key: jax.Array | None = None,
) -> jax.Array:
    """Refine reasoning embedding via Langevin dynamics toward lower energy.

    **Researcher summary:**
        l(s+1) = l(s) - η∇E(l(s)) + √(2η)ε — pushes reasoning toward
        coherent (low-energy) regions in embedding space.

    Spec: REQ-INFER-011
    """
    if key is None:
        key = jrandom.PRNGKey(0)

    x = chain_embedding
    for _ in range(n_langevin_steps):
        key, subkey = jrandom.split(key)
        grad = model.grad_energy(x)
        noise = jrandom.normal(subkey, x.shape) * noise_scale
        x = x - step_size * grad + noise

    return x


# Template-based training data generation

_COHERENT_TEMPLATES = [
    "2 + 3 = 5, therefore 5 * 2 = 10",
    "if x = 4, then x + 1 = 5",
    "3 * 3 = 9, and 9 - 4 = 5",
    "10 / 2 = 5, so 5 + 5 = 10",
    "7 - 3 = 4, and 4 * 2 = 8",
]

_INCOHERENT_TEMPLATES = [
    "2 + 3 = 7, therefore 5 * 2 = 3",
    "if x = 4, then x + 1 = 9",
    "3 * 3 = 6, and 9 - 4 = 12",
    "10 / 2 = 3, so 5 + 5 = 7",
    "7 - 3 = 1, and 4 * 2 = 15",
]


def generate_reasoning_training_data(
    n_samples: int = 100,
    vocab_size: int = 256,
    seed: int = 42,
) -> tuple[jax.Array, jax.Array]:
    """Generate (coherent, incoherent) reasoning embedding pairs for NCE.

    Spec: REQ-INFER-011
    """
    rng = random.Random(seed)
    coherent_list: list[jax.Array] = []
    incoherent_list: list[jax.Array] = []

    for _ in range(n_samples):
        # Pick a random template and add minor variation
        c_text = rng.choice(_COHERENT_TEMPLATES) + f" step {rng.randint(0, 100)}"
        i_text = rng.choice(_INCOHERENT_TEMPLATES) + f" step {rng.randint(0, 100)}"
        coherent_list.append(text_to_reasoning_embedding(c_text, vocab_size))
        incoherent_list.append(text_to_reasoning_embedding(i_text, vocab_size))

    return jnp.stack(coherent_list), jnp.stack(incoherent_list)


def train_reasoning_energy(
    coherent_embeddings: jax.Array,
    incoherent_embeddings: jax.Array,
    config: ReasoningVerifierConfig | None = None,
) -> GibbsModel:
    """Train reasoning energy model via NCE.

    **Researcher summary:**
        NCE on (coherent, incoherent) pairs. Same training loop as
        train_sat_verifier and train_code_verifier.

    Spec: REQ-INFER-011, SCENARIO-INFER-012
    """
    if config is None:
        config = ReasoningVerifierConfig()

    key = jrandom.PRNGKey(config.seed)
    key, model_key = jrandom.split(key)

    gibbs_config = GibbsConfig(
        input_dim=config.vocab_size,
        hidden_dims=config.hidden_dims,
        activation="silu",
    )
    model = GibbsModel(gibbs_config, key=model_key)

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
        result = nce_loss(model, coherent_embeddings, incoherent_embeddings)
        set_params(model, old)
        return result

    params = get_params(model)
    for epoch in range(config.n_epochs):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(
            lambda p, g: p - config.learning_rate * g,
            params,
            grads,
        )

    set_params(model, params)
    return model


def verify_reasoning_chain(
    reasoning_text: str,
    model: GibbsModel,
    vocab_size: int = 256,
    threshold: float = 1.0,
) -> ReasoningEnergyResult:
    """Full pipeline: embed → compute energy → classify coherence.

    Spec: REQ-INFER-011, SCENARIO-INFER-012
    """
    embedding = text_to_reasoning_embedding(reasoning_text, vocab_size)
    energy = compute_reasoning_energy(model, embedding)
    return ReasoningEnergyResult(
        chain_energy=energy,
        is_coherent=energy < threshold,
        threshold=threshold,
    )
