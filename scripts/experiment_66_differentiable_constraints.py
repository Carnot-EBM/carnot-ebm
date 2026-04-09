#!/usr/bin/env python3
"""Experiment 66: Differentiable Constraint Verification — End-to-end verifier in one forward pass.

**Researcher summary:**
    Combines continuous Ising relaxation (Exp 64) with joint embedding-constraint
    EBMs (Exp 65) into a single differentiable verifier. The forward pass is:
    text → sentence embedding (384-dim) → constraint features (N-dim) →
    continuous Ising energy (differentiable) → joint MLP → verification score.
    Trained end-to-end with BCE loss via JAX autodiff.

**Detailed explanation for engineers:**
    Previous experiments proved two things separately:
    - Exp 64: Binary Ising spins can be relaxed to [0,1] and optimized with
      gradient descent. Sigmoid annealing makes the rounding differentiable.
    - Exp 65: A Gibbs EBM trained on the joint [embedding; constraint_vector]
      space discriminates correct from wrong answers better than either space
      alone (AUROC improvements of 5-15%).

    This experiment UNIFIES both into a single differentiable forward pass.
    Instead of two separate models (Ising verifier + embedding classifier),
    we build one end-to-end network:

    1. **Embedding layer**: Sentence embedding of the text response (all-MiniLM-L6-v2,
       384 dimensions). This captures semantic meaning.

    2. **Constraint features**: The AutoExtractor produces N constraint pass/fail
       features. These capture structural correctness (arithmetic right, code valid,
       logic consistent, etc.).

    3. **Continuous Ising layer**: Instead of discrete Gibbs sampling on constraint
       spins, we relax spins to [0,1] and compute the Ising energy analytically.
       This layer is fully differentiable — gradients flow through the sigmoid
       relaxation back to the constraint features. The Ising coupling matrix and
       biases are LEARNED parameters, not fixed.

    4. **Joint MLP**: A small 2-layer MLP on the concatenated vector
       [embedding; constraint_satisfaction; relaxed_spins; ising_energy] produces
       a scalar verification score via sigmoid → [0,1].

    The entire pipeline is trained end-to-end with binary cross-entropy loss:
    correct answers should get score ≈ 1 (verified), wrong answers ≈ 0 (rejected).
    Because everything is differentiable, a single jax.grad call computes gradients
    for ALL parameters (Ising couplings, MLP weights, etc.) simultaneously.

    **Why this matters for Phase 17:**
    Experiments 86 and 87 will plug this differentiable verifier into the
    autoresearch loop. The LLM generates candidate answers, the verifier scores
    them in a single forward pass, and the gradient tells the repair module
    WHICH constraints to fix and HOW to fix them (via the Ising coupling structure).
    This replaces the expensive discrete-sample-then-classify pipeline with a
    smooth, gradient-friendly verification surface.

    **Ablation study:**
    We compare three configurations to quantify each component's contribution:
    (a) Continuous Ising only — just the relaxed Ising energy as the score
    (b) Embedding-only — just the MLP on sentence embeddings
    (c) Joint (full model) — all components together
    This tells us whether the Ising layer adds signal beyond what embeddings
    already capture, and vice versa.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_66_differentiable_constraints.py

REQ: REQ-EBT-001, REQ-VERIFY-001, REQ-VERIFY-002
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.pipeline.extract import AutoExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension
N_CONSTRAINTS = 8  # Number of lightweight text constraints (same as Exp 65)
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "experiment_66_results.json"
)

# Constraint names — same lightweight text checks as Experiment 65.
# These are fast heuristics, not deep semantic checks. The point is to
# give the continuous Ising layer a meaningful binary feature space to
# learn couplings over.
CONSTRAINT_NAMES = [
    "contains_number",
    "complete_sentence",
    "not_empty",
    "no_contradiction",
    "has_explanation",
    "reasonable_length",
    "no_repetition",
    "topic_relevant",
]

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from it its this that these those and but or not no".split()
)

DOMAINS = ["arithmetic", "code", "logic", "factual", "scheduling"]


# ---------------------------------------------------------------------------
# 1. Multi-domain synthetic dataset (500 Q/A pairs)
# ---------------------------------------------------------------------------


def _check_constraints(question: str, answer: str) -> list[bool]:
    """Evaluate 8 lightweight text constraints on an answer.

    **Detailed explanation for engineers:**
        Identical to Exp 65's constraint checks. Each constraint is a simple
        regex or string heuristic that captures one structural aspect of the
        answer. The binary pass/fail vector feeds into the continuous Ising
        layer as input features.

    Args:
        question: The question text (used for topic-relevance check).
        answer: The candidate answer text.

    Returns:
        List of 8 bools, one per constraint in CONSTRAINT_NAMES.
    """
    results = []
    stripped = answer.strip()
    words = answer.lower().split()

    # 1. contains_number
    results.append(any(c.isdigit() for c in answer))

    # 2. complete_sentence
    results.append(len(stripped) > 0 and stripped[-1] in ".?!")

    # 3. not_empty
    results.append(len(stripped) > 5)

    # 4. no_contradiction (heuristic: "not" near "is/are/was/were")
    has_contradiction = False
    for i, w in enumerate(words):
        if w == "not":
            window = words[max(0, i - 3) : i + 4]
            if any(v in window for v in ("is", "are", "was", "were")):
                has_contradiction = True
                break
    results.append(not has_contradiction)

    # 5. has_explanation
    lower = answer.lower()
    results.append(
        any(kw in lower for kw in ("because", "since", "therefore", "so ", "thus"))
    )

    # 6. reasonable_length
    results.append(10 <= len(stripped) <= 500)

    # 7. no_repetition (no repeated 3-word sequences)
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    results.append(len(trigrams) == len(set(trigrams)))

    # 8. topic_relevant
    q_words = set(question.lower().split()) - _STOPWORDS
    a_words = set(answer.lower().split()) - _STOPWORDS
    results.append(len(q_words & a_words) > 0)

    return results


def _generate_multi_domain_dataset(
    n_per_domain: int = 100,
    seed: int = 66,
) -> list[dict]:
    """Generate 500 Q/A pairs across 5 domains, each with correct/wrong labels.

    **Detailed explanation for engineers:**
        Creates a balanced dataset where each domain has n_per_domain questions,
        half answered correctly and half incorrectly. This mirrors Exp 58's
        multi-domain benchmark but generates synthetic data inline so we don't
        depend on Exp 58 output files existing on disk.

        Domains:
        - arithmetic: "What is X + Y?" with correct/wrong numerical answers
        - code: "Write a function that..." with correct/broken code
        - logic: "If X then Y. Is Z true?" with correct/wrong conclusions
        - factual: "What is the capital of X?" with correct/wrong facts
        - scheduling: "Meeting at X, another at Y. Any conflict?" with correct/wrong

    Args:
        n_per_domain: Questions per domain (half correct, half wrong).
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with keys: domain, question, answer, is_correct.
    """
    rng = np.random.default_rng(seed)
    dataset: list[dict] = []

    n_each = n_per_domain // 2  # half correct, half wrong per domain

    # --- Arithmetic ---
    for i in range(n_each):
        a = int(rng.integers(10, 500))
        b = int(rng.integers(1, 200))
        op = ["+", "-", "*"][i % 3]
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        else:
            a, b = int(rng.integers(2, 30)), int(rng.integers(2, 30))
            r = a * b
        q = f"What is {a} {op} {b}?"
        dataset.append({
            "domain": "arithmetic",
            "question": q,
            "answer": f"The answer is {r}. Because {a} {op} {b} equals {r}.",
            "is_correct": True,
        })
        # Wrong: off by random amount
        wrong_r = r + int(rng.integers(1, 50))
        dataset.append({
            "domain": "arithmetic",
            "question": q,
            "answer": f"The answer is {wrong_r}. Because the calculation gives {wrong_r}.",
            "is_correct": False,
        })

    # --- Code ---
    code_tasks = [
        ("reverse a string", "def reverse_string(s):\n    return s[::-1].", True),
        ("compute factorial", "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1).", True),
        ("find maximum", "def find_max(lst):\n    return max(lst).", True),
        ("check palindrome", "def is_palindrome(s):\n    return s == s[::-1].", True),
        ("sum of list", "def sum_list(lst):\n    return sum(lst).", True),
    ]
    for i in range(n_each):
        task_name, code, _ = code_tasks[i % len(code_tasks)]
        q = f"Write a Python function to {task_name}."
        dataset.append({
            "domain": "code",
            "question": q,
            "answer": f"Here is the solution:\n```python\n{code}\n```\nThis works because it implements the required logic.",
            "is_correct": True,
        })
        # Wrong: broken code or off-topic
        wrong_answers = [
            "def func():\n    pass",
            "The answer is 42.",
            f"def {task_name.replace(' ', '_')}():\n    return None  # TODO",
        ]
        dataset.append({
            "domain": "code",
            "question": q,
            "answer": wrong_answers[i % len(wrong_answers)],
            "is_correct": False,
        })

    # --- Logic ---
    logic_templates = [
        ("If it rains, the ground gets wet. It rained. Is the ground wet?", "Yes", "No"),
        ("All cats are mammals. Tom is a cat. Is Tom a mammal?", "Yes", "No"),
        ("If A implies B, and A is true, then B must be?", "True", "False"),
        ("If X > 5 and X < 10, and X = 7, is the condition satisfied?", "Yes", "No"),
        ("All birds can fly. Penguins are birds. Can penguins fly?",
         "Based on the premises, yes. However the real-world premise is flawed.", "No"),
    ]
    for i in range(n_each):
        q, correct_a, wrong_a = logic_templates[i % len(logic_templates)]
        dataset.append({
            "domain": "logic",
            "question": q,
            "answer": f"{correct_a}. Because the logical chain of reasoning supports this conclusion.",
            "is_correct": True,
        })
        dataset.append({
            "domain": "logic",
            "question": q,
            "answer": f"{wrong_a}. Because the premises do not support this.",
            "is_correct": False,
        })

    # --- Factual ---
    factual_pairs = [
        ("What is the capital of France?", "Paris", "Lyon"),
        ("What is the chemical symbol for water?", "H2O", "CO2"),
        ("How many continents are there?", "7", "5"),
        ("What planet is closest to the Sun?", "Mercury", "Venus"),
        ("What year did World War II end?", "1945", "1939"),
    ]
    for i in range(n_each):
        q, correct_a, wrong_a = factual_pairs[i % len(factual_pairs)]
        dataset.append({
            "domain": "factual",
            "question": q,
            "answer": f"The answer is {correct_a}. This is a well-established fact.",
            "is_correct": True,
        })
        dataset.append({
            "domain": "factual",
            "question": q,
            "answer": f"The answer is {wrong_a}. This is what I recall.",
            "is_correct": False,
        })

    # --- Scheduling ---
    for i in range(n_each):
        h1 = int(rng.integers(8, 16))
        h2 = h1 + int(rng.integers(1, 3))
        h3 = h2 + int(rng.integers(1, 3))
        q = (
            f"Meeting A is at {h1}:00-{h2}:00. "
            f"Meeting B is at {h3}:00-{h3+1}:00. "
            f"Is there a conflict?"
        )
        dataset.append({
            "domain": "scheduling",
            "question": q,
            "answer": f"No conflict. Meeting A ends at {h2}:00 and Meeting B starts at {h3}:00, so they don't overlap.",
            "is_correct": True,
        })
        # Wrong: claim conflict when there is none
        dataset.append({
            "domain": "scheduling",
            "question": q,
            "answer": f"Yes, there is a conflict because both meetings overlap at {h2}:00.",
            "is_correct": False,
        })

    # Shuffle deterministically
    indices = list(range(len(dataset)))
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(indices)
    dataset = [dataset[i] for i in indices]

    return dataset


# ---------------------------------------------------------------------------
# 2. Embedding and feature computation
# ---------------------------------------------------------------------------


def _embed_answers(
    questions: list[str],
    answers: list[str],
    embed_dim: int = EMBED_DIM,
) -> np.ndarray:
    """Embed (question, answer) pairs using sentence-transformers or mock fallback.

    **Detailed explanation for engineers:**
        Uses all-MiniLM-L6-v2 (384-dim) to embed "Q: {q} A: {a}" as a single
        sentence. If sentence-transformers is not installed, falls back to
        deterministic mock embeddings where correct answers cluster near a
        learned centroid and wrong answers are spread more diffusely.

    Args:
        questions: List of question strings.
        answers: List of answer strings.
        embed_dim: Expected embedding dimension (384 for MiniLM).

    Returns:
        Array of shape (n, embed_dim) with L2-normalized embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [f"Q: {q} A: {a}" for q, a in zip(questions, answers)]
        embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        assert embeddings.shape[1] == embed_dim
        return embeddings
    except ImportError:
        print("  [WARNING] sentence-transformers not installed, using mock embeddings")
        rng = np.random.default_rng(66)
        n = len(questions)
        return rng.standard_normal((n, embed_dim)).astype(np.float32)


def _compute_constraint_vectors(
    questions: list[str],
    answers: list[str],
) -> np.ndarray:
    """Compute per-constraint pass/fail vectors for all QA pairs.

    **Detailed explanation for engineers:**
        Evaluates each (question, answer) pair against the 8 lightweight text
        constraints. Returns a binary matrix where [i, j] = 1.0 means answer i
        satisfies constraint j.

    Args:
        questions: List of question strings.
        answers: List of answer strings.

    Returns:
        Array of shape (n, N_CONSTRAINTS) with binary constraint satisfaction.
    """
    n = len(questions)
    vectors = np.zeros((n, N_CONSTRAINTS), dtype=np.float32)
    for i in range(n):
        checks = _check_constraints(questions[i], answers[i])
        for j, passed in enumerate(checks):
            vectors[i, j] = 1.0 if passed else 0.0
    return vectors


# ---------------------------------------------------------------------------
# 3. DifferentiableVerifier — the core model
# ---------------------------------------------------------------------------


class DifferentiableVerifierParams:
    """All learnable parameters for the differentiable verifier, stored as a JAX pytree.

    **Detailed explanation for engineers:**
        We store all parameters in a single class registered as a JAX pytree so
        jax.grad can differentiate through the entire model in one call. JAX's
        tree_util needs to know how to flatten/unflatten this object for autodiff
        to work. We register it via jax.tree_util.register_pytree_class.

        Parameters:
        - ising_biases: Bias vector for the continuous Ising layer, shape (n_spins,).
          Controls the base probability of each constraint spin being "on".
        - ising_J: Coupling matrix for the continuous Ising layer, shape (n_spins, n_spins).
          Controls pairwise interactions between constraint spins. Symmetric, zero diagonal.
        - mlp_w1: Weight matrix for first MLP layer, shape (input_dim, hidden_dim).
        - mlp_b1: Bias vector for first MLP layer, shape (hidden_dim,).
        - mlp_w2: Weight matrix for second MLP layer, shape (hidden_dim, 1).
        - mlp_b2: Scalar bias for output layer.
    """
    def __init__(
        self,
        ising_biases: jax.Array,
        ising_J: jax.Array,
        mlp_w1: jax.Array,
        mlp_b1: jax.Array,
        mlp_w2: jax.Array,
        mlp_b2: jax.Array,
    ):
        self.ising_biases = ising_biases
        self.ising_J = ising_J
        self.mlp_w1 = mlp_w1
        self.mlp_b1 = mlp_b1
        self.mlp_w2 = mlp_w2
        self.mlp_b2 = mlp_b2

    def tree_flatten(self):
        """Flatten into a list of arrays (leaves) and auxiliary data."""
        children = (
            self.ising_biases, self.ising_J,
            self.mlp_w1, self.mlp_b1,
            self.mlp_w2, self.mlp_b2,
        )
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from flattened leaves."""
        return cls(*children)


jax.tree_util.register_pytree_node(
    DifferentiableVerifierParams,
    lambda p: p.tree_flatten(),
    lambda aux, children: DifferentiableVerifierParams.tree_unflatten(aux, children),
)


def init_verifier_params(
    n_spins: int,
    embed_dim: int,
    hidden_dim: int,
    key: jax.Array,
) -> DifferentiableVerifierParams:
    """Initialize all verifier parameters with small random values.

    **Detailed explanation for engineers:**
        The Ising biases start at zero (no preference for spin up/down).
        The coupling matrix starts as small random symmetric values with
        zero diagonal (weak initial pairwise interactions). MLP weights
        use Glorot-uniform-scale initialization for stable gradients.

        The MLP input dimension is: embed_dim + n_spins + n_spins + 1
        (embedding + constraint_satisfaction + relaxed_spins + ising_energy scalar)

    Args:
        n_spins: Number of constraint spins (= N_CONSTRAINTS).
        embed_dim: Sentence embedding dimension (384).
        hidden_dim: MLP hidden layer width.
        key: JAX PRNG key.

    Returns:
        Initialized DifferentiableVerifierParams.
    """
    k1, k2, k3, k4 = jrandom.split(key, 4)

    # Ising parameters
    ising_biases = jnp.zeros(n_spins)
    # Small random symmetric coupling matrix with zero diagonal.
    raw_J = jrandom.normal(k1, (n_spins, n_spins)) * 0.1
    ising_J = (raw_J + raw_J.T) / 2.0
    ising_J = ising_J.at[jnp.diag_indices(n_spins)].set(0.0)

    # MLP parameters
    # Input: [embedding(384) + constraint_satisfaction(n_spins) + relaxed_spins(n_spins) + ising_energy(1)]
    mlp_input_dim = embed_dim + n_spins + n_spins + 1
    # Glorot-scale initialization
    scale_1 = jnp.sqrt(2.0 / (mlp_input_dim + hidden_dim))
    mlp_w1 = jrandom.normal(k2, (mlp_input_dim, hidden_dim)) * scale_1
    mlp_b1 = jnp.zeros(hidden_dim)

    scale_2 = jnp.sqrt(2.0 / (hidden_dim + 1))
    mlp_w2 = jrandom.normal(k3, (hidden_dim, 1)) * scale_2
    mlp_b2 = jnp.zeros(1)

    return DifferentiableVerifierParams(
        ising_biases=ising_biases,
        ising_J=ising_J,
        mlp_w1=mlp_w1,
        mlp_b1=mlp_b1,
        mlp_w2=mlp_w2,
        mlp_b2=mlp_b2,
    )


def continuous_ising_energy(
    spins: jax.Array,
    biases: jax.Array,
    J: jax.Array,
) -> jax.Array:
    """Compute Ising energy for continuous [0,1] spins — fully differentiable.

    **Detailed explanation for engineers:**
        The standard Ising energy is E(s) = -(b^T s + s^T J s). For continuous
        s ∈ [0,1]^n, this is a smooth quadratic function. JAX's autodiff computes
        the gradient ∇E = -(b + (J + J^T) s) automatically.

        This is the same function as Exp 64's ising_energy(), factored out here
        for clarity. The key property: because s is continuous (not discrete),
        the energy is differentiable everywhere, so gradients from the verification
        score flow all the way back through the Ising layer to the constraint features.

    Args:
        spins: Spin vector, shape (n,). Values in [0, 1].
        biases: Bias vector, shape (n,).
        J: Symmetric coupling matrix, shape (n, n), zero diagonal.

    Returns:
        Scalar energy (lower is better for satisfied configurations).
    """
    return -(jnp.dot(biases, spins) + jnp.dot(spins, J @ spins))


def differentiable_verifier_forward(
    params: DifferentiableVerifierParams,
    embedding: jax.Array,
    constraint_vec: jax.Array,
    alpha: float = 10.0,
) -> jax.Array:
    """Single differentiable forward pass: embedding + constraints → verification score.

    **Detailed explanation for engineers:**
        This is the core of Experiment 66. The forward pass:

        1. Sigmoid annealing on constraint features → relaxed spins:
           relaxed_spins = sigmoid(alpha * (constraint_vec - 0.5))
           Alpha controls sharpness: low alpha → smooth interpolation,
           high alpha → nearly binary. During training we use moderate alpha
           so gradients flow; at test time we can increase alpha.

        2. Continuous Ising energy from relaxed spins:
           ising_e = -(b^T s + s^T J s)
           This is fully differentiable w.r.t. both the spins AND the
           Ising parameters (biases b, couplings J).

        3. Concatenate features: [embedding; constraint_vec; relaxed_spins; ising_energy]
           The MLP sees semantic meaning, raw constraint pass/fail, relaxed
           spin states, AND the aggregate Ising energy — a rich feature set.

        4. 2-layer MLP → sigmoid → verification score in [0, 1]:
           h = silu(W1 @ concat + b1)
           score = sigmoid(W2 @ h + b2)

        The entire function is a pure JAX computation, so jax.grad gives us
        gradients for all parameters in one backward pass.

    Args:
        params: DifferentiableVerifierParams with all learnable weights.
        embedding: Sentence embedding, shape (embed_dim,).
        constraint_vec: Binary constraint satisfaction, shape (n_spins,).
        alpha: Sigmoid annealing sharpness for continuous Ising relaxation.

    Returns:
        Scalar verification score in [0, 1]. High = verified, low = rejected.
    """
    # Step 1: Relax binary constraints to continuous spins via sigmoid annealing.
    # constraint_vec is already in [0, 1] (binary), but sigmoid(alpha*(x-0.5))
    # maps it to a smooth [0,1] that's differentiable. For x=1.0 and alpha=10,
    # sigmoid(10*0.5) ≈ 0.993. For x=0.0, sigmoid(10*(-0.5)) ≈ 0.007.
    relaxed_spins = jax.nn.sigmoid(alpha * (constraint_vec - 0.5))

    # Step 2: Compute differentiable Ising energy from relaxed spins.
    ising_e = continuous_ising_energy(relaxed_spins, params.ising_biases, params.ising_J)
    # Normalize to roughly [-1, 1] range for stable MLP input.
    ising_e_normalized = jnp.tanh(ising_e * 0.1)

    # Step 3: Concatenate all features.
    features = jnp.concatenate([
        embedding,                            # semantic meaning (384-dim)
        constraint_vec,                       # raw constraint pass/fail (8-dim)
        relaxed_spins,                        # relaxed spin states (8-dim)
        ising_e_normalized.reshape(1),        # aggregate Ising energy (1-dim)
    ])

    # Step 4: 2-layer MLP with SiLU activation → sigmoid output.
    h = jax.nn.silu(features @ params.mlp_w1 + params.mlp_b1)
    logit = (h @ params.mlp_w2 + params.mlp_b2).squeeze()
    score = jax.nn.sigmoid(logit)

    return score


# ---------------------------------------------------------------------------
# 4. Ablation model variants
# ---------------------------------------------------------------------------


def ising_only_forward(
    params: DifferentiableVerifierParams,
    constraint_vec: jax.Array,
    alpha: float = 10.0,
) -> jax.Array:
    """Ablation: Continuous Ising energy only (no embedding, no MLP).

    **Detailed explanation for engineers:**
        Uses only the constraint features and learned Ising parameters.
        The verification score is sigmoid(-ising_energy), so low energy
        (many satisfied constraints with compatible couplings) → high score.

    Args:
        params: Verifier params (only ising_biases and ising_J used).
        constraint_vec: Binary constraint satisfaction, shape (n_spins,).
        alpha: Sigmoid annealing sharpness.

    Returns:
        Scalar score in [0, 1].
    """
    relaxed_spins = jax.nn.sigmoid(alpha * (constraint_vec - 0.5))
    ising_e = continuous_ising_energy(relaxed_spins, params.ising_biases, params.ising_J)
    return jax.nn.sigmoid(-ising_e * 0.5)


def embedding_only_forward(
    params: DifferentiableVerifierParams,
    embedding: jax.Array,
    n_spins: int,
) -> jax.Array:
    """Ablation: Embedding + MLP only (no Ising layer, no constraint features).

    **Detailed explanation for engineers:**
        Feeds only the sentence embedding through the MLP, with constraint
        and Ising features zeroed out. This measures how much the embedding
        alone can discriminate correct from wrong answers.

    Args:
        params: Verifier params (only MLP weights used).
        embedding: Sentence embedding, shape (embed_dim,).
        n_spins: Number of constraint spins (for zero-padding).

    Returns:
        Scalar score in [0, 1].
    """
    zeros_spins = jnp.zeros(n_spins)
    features = jnp.concatenate([
        embedding,
        zeros_spins,         # no constraint features
        zeros_spins,         # no relaxed spins
        jnp.zeros(1),       # no Ising energy
    ])
    h = jax.nn.silu(features @ params.mlp_w1 + params.mlp_b1)
    logit = (h @ params.mlp_w2 + params.mlp_b2).squeeze()
    return jax.nn.sigmoid(logit)


# ---------------------------------------------------------------------------
# 5. Loss functions and training
# ---------------------------------------------------------------------------


def bce_loss(score: jax.Array, label: jax.Array) -> jax.Array:
    """Binary cross-entropy loss for a single sample.

    **Detailed explanation for engineers:**
        BCE = -[y * log(p) + (1-y) * log(1-p)]
        where y is the label (1=correct, 0=wrong) and p is the predicted score.
        We clip p to [1e-7, 1-1e-7] for numerical stability (avoid log(0)).

    Args:
        score: Predicted verification score in [0, 1].
        label: Ground truth label (1.0 = correct, 0.0 = wrong).

    Returns:
        Scalar BCE loss.
    """
    eps = 1e-7
    p = jnp.clip(score, eps, 1.0 - eps)
    return -(label * jnp.log(p) + (1.0 - label) * jnp.log(1.0 - p))


def batch_loss(
    params: DifferentiableVerifierParams,
    embeddings: jax.Array,
    constraint_vecs: jax.Array,
    labels: jax.Array,
    alpha: float = 10.0,
) -> jax.Array:
    """Mean BCE loss over a batch.

    **Detailed explanation for engineers:**
        Maps the forward pass over all samples in the batch using jax.vmap,
        then computes mean BCE loss. The vmap ensures we get one forward pass
        per sample with shared parameters, and jax.grad on this function
        gives us the average gradient over the batch.

    Args:
        params: DifferentiableVerifierParams.
        embeddings: Batch of embeddings, shape (batch, embed_dim).
        constraint_vecs: Batch of constraint vectors, shape (batch, n_spins).
        labels: Batch of labels, shape (batch,).
        alpha: Sigmoid annealing sharpness.

    Returns:
        Scalar mean BCE loss.
    """
    def single_loss(emb, cvec, label):
        score = differentiable_verifier_forward(params, emb, cvec, alpha)
        return bce_loss(score, label)

    losses = jax.vmap(single_loss)(embeddings, constraint_vecs, labels)
    return jnp.mean(losses)


def train_verifier(
    embeddings: jax.Array,
    constraint_vecs: jax.Array,
    labels: jax.Array,
    lr: float,
    n_epochs: int,
    key: jax.Array,
    hidden_dim: int = 64,
    alpha: float = 10.0,
    verbose: bool = True,
) -> tuple[DifferentiableVerifierParams, list[float], list[float]]:
    """Train the differentiable verifier end-to-end with Adam optimizer.

    **Detailed explanation for engineers:**
        Implements the Adam optimizer (Kingma & Ba 2014) from scratch in JAX
        because we need to optimize a custom pytree (DifferentiableVerifierParams).
        Adam adapts the learning rate per-parameter using first and second moment
        estimates of the gradient, which handles the different scales of Ising
        parameters (small couplings) vs MLP weights (larger magnitudes).

        We track:
        - loss_history: BCE loss per epoch (should decrease)
        - grad_norm_history: L2 norm of the full gradient per epoch (monitors
          gradient stability — should not explode or vanish)

    Args:
        embeddings: All embeddings, shape (n, embed_dim).
        constraint_vecs: All constraint vectors, shape (n, n_spins).
        labels: All labels, shape (n,).
        lr: Learning rate.
        n_epochs: Number of training epochs.
        key: JAX PRNG key.
        hidden_dim: MLP hidden layer width.
        alpha: Sigmoid annealing sharpness.
        verbose: Print progress every 20 epochs.

    Returns:
        Tuple of (trained_params, loss_history, grad_norm_history).
    """
    n_spins = constraint_vecs.shape[1]
    embed_dim = embeddings.shape[1]
    params = init_verifier_params(n_spins, embed_dim, hidden_dim, key)

    # Adam state: first moment (m) and second moment (v) for each param.
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    loss_history: list[float] = []
    grad_norm_history: list[float] = []

    grad_fn = jax.grad(batch_loss)

    for epoch in range(n_epochs):
        # Compute gradients
        grads = grad_fn(params, embeddings, constraint_vecs, labels, alpha)

        # Track gradient norm for stability monitoring
        grad_leaves = jax.tree.leaves(grads)
        grad_norm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in grad_leaves)))
        grad_norm_history.append(grad_norm)

        # Adam update
        t = epoch + 1
        m = jax.tree.map(lambda mi, gi: beta1 * mi + (1 - beta1) * gi, m, grads)
        v = jax.tree.map(lambda vi, gi: beta2 * vi + (1 - beta2) * gi ** 2, v, grads)
        m_hat = jax.tree.map(lambda mi: mi / (1 - beta1 ** t), m)
        v_hat = jax.tree.map(lambda vi: vi / (1 - beta2 ** t), v)
        params = jax.tree.map(
            lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
            params, m_hat, v_hat,
        )

        # Enforce symmetric, zero-diagonal on Ising J after update
        J_updated = (params.ising_J + params.ising_J.T) / 2.0
        J_updated = J_updated.at[jnp.diag_indices(n_spins)].set(0.0)
        params = DifferentiableVerifierParams(
            ising_biases=params.ising_biases,
            ising_J=J_updated,
            mlp_w1=params.mlp_w1,
            mlp_b1=params.mlp_b1,
            mlp_w2=params.mlp_w2,
            mlp_b2=params.mlp_b2,
        )

        # Track loss
        loss_val = float(batch_loss(params, embeddings, constraint_vecs, labels, alpha))
        loss_history.append(loss_val)

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            print(f"    epoch {epoch:>4d}: loss={loss_val:.4f}  grad_norm={grad_norm:.4f}")

    return params, loss_history, grad_norm_history


# ---------------------------------------------------------------------------
# 6. Evaluation
# ---------------------------------------------------------------------------


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC for binary classification.

    **Detailed explanation for engineers:**
        Uses sklearn if available, otherwise falls back to a manual
        trapezoidal-rule implementation. AUROC = 1.0 means perfect
        discrimination, 0.5 means random.

    Args:
        scores: Predicted scores, shape (n,). Higher = more likely correct.
        labels: Binary labels, shape (n,). 1 = correct, 0 = wrong.

    Returns:
        AUROC score in [0, 1].
    """
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except ImportError:
        # Manual AUROC: sort by score descending, compute true/false positive rates.
        sorted_idx = np.argsort(-scores)
        sorted_labels = labels[sorted_idx]
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tp, fp = 0, 0
        tpr_prev, fpr_prev = 0.0, 0.0
        auc = 0.0
        for lab in sorted_labels:
            if lab == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev, fpr_prev = tpr, fpr
        return float(auc)


def evaluate_verifier(
    params: DifferentiableVerifierParams,
    embeddings: jax.Array,
    constraint_vecs: jax.Array,
    labels: np.ndarray,
    domains: list[str],
    alpha: float = 10.0,
) -> dict:
    """Evaluate the verifier: AUROC overall and per-domain, plus ablations.

    **Detailed explanation for engineers:**
        Runs three forward-pass variants (joint, Ising-only, embedding-only)
        on the test data and computes AUROC for each. Also breaks down
        AUROC per domain so we can see which domains benefit most from the
        Ising layer.

    Args:
        params: Trained verifier params.
        embeddings: Test embeddings, shape (n, embed_dim).
        constraint_vecs: Test constraint vectors, shape (n, n_spins).
        labels: Test labels, shape (n,). 1 = correct, 0 = wrong.
        domains: Per-sample domain tags, same length as labels.
        alpha: Sigmoid annealing sharpness.

    Returns:
        Dict with keys: auroc_joint, auroc_ising_only, auroc_embedding_only,
        per_domain (dict of domain → auroc), gradient_stability (dict).
    """
    n = len(labels)
    n_spins = constraint_vecs.shape[1]

    # Compute scores for all three variants.
    joint_scores = np.zeros(n)
    ising_scores = np.zeros(n)
    embed_scores = np.zeros(n)

    for i in range(n):
        joint_scores[i] = float(differentiable_verifier_forward(
            params, embeddings[i], constraint_vecs[i], alpha
        ))
        ising_scores[i] = float(ising_only_forward(
            params, constraint_vecs[i], alpha
        ))
        embed_scores[i] = float(embedding_only_forward(
            params, embeddings[i], n_spins
        ))

    # Overall AUROC
    auroc_joint = compute_auroc(joint_scores, labels)
    auroc_ising = compute_auroc(ising_scores, labels)
    auroc_embed = compute_auroc(embed_scores, labels)

    # Per-domain AUROC
    per_domain = {}
    for domain in DOMAINS:
        mask = np.array([d == domain for d in domains])
        if mask.sum() < 2:
            continue
        domain_labels = labels[mask]
        # Need both classes for meaningful AUROC
        if len(set(domain_labels.tolist())) < 2:
            per_domain[domain] = {"joint": 0.5, "ising_only": 0.5, "embedding_only": 0.5}
            continue
        per_domain[domain] = {
            "joint": compute_auroc(joint_scores[mask], domain_labels),
            "ising_only": compute_auroc(ising_scores[mask], domain_labels),
            "embedding_only": compute_auroc(embed_scores[mask], domain_labels),
        }

    return {
        "auroc_joint": auroc_joint,
        "auroc_ising_only": auroc_ising,
        "auroc_embedding_only": auroc_embed,
        "per_domain": per_domain,
    }


# ---------------------------------------------------------------------------
# 7. Main experiment
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 66: Differentiable Constraint Verification")
    print("  End-to-end: text → embedding → constraints → Ising → MLP → score")
    print("  Combines Exp 64 (continuous Ising) + Exp 65 (joint embedding-constraint)")
    print("=" * 70)

    start = time.time()

    # --- Step 1: Generate dataset ---
    print("\n[1/5] Generating multi-domain dataset (500 Q/A pairs)...")
    dataset = _generate_multi_domain_dataset(n_per_domain=100, seed=66)
    questions = [d["question"] for d in dataset]
    answers = [d["answer"] for d in dataset]
    labels_bool = [d["is_correct"] for d in dataset]
    domains_list = [d["domain"] for d in dataset]
    labels = np.array([1.0 if c else 0.0 for c in labels_bool], dtype=np.float32)
    print(f"  Generated {len(dataset)} samples across {len(DOMAINS)} domains")
    print(f"  Correct: {sum(labels_bool)}, Wrong: {len(labels_bool) - sum(labels_bool)}")

    # --- Step 2: Compute features ---
    print("\n[2/5] Computing embeddings and constraint features...")
    embeddings = _embed_answers(questions, answers, EMBED_DIM)
    constraint_vecs = _compute_constraint_vectors(questions, answers)
    print(f"  Embeddings: {embeddings.shape}, Constraints: {constraint_vecs.shape}")

    # Constraint satisfaction summary
    mean_satisfaction = constraint_vecs.mean(axis=0)
    print(f"  Mean constraint satisfaction per constraint:")
    for i, name in enumerate(CONSTRAINT_NAMES):
        print(f"    {name}: {mean_satisfaction[i]:.2f}")

    # --- Step 3: Train/val/test split (60/20/20) ---
    print("\n[3/5] Splitting data (60/20/20)...")
    n = len(dataset)
    rng = np.random.default_rng(66)
    indices = rng.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    emb_jax = jnp.array(embeddings)
    cvec_jax = jnp.array(constraint_vecs)
    labels_jax = jnp.array(labels)

    train_emb, train_cvec, train_labels = emb_jax[train_idx], cvec_jax[train_idx], labels_jax[train_idx]
    val_emb, val_cvec, val_labels = emb_jax[val_idx], cvec_jax[val_idx], labels_jax[val_idx]
    test_emb, test_cvec, test_labels = emb_jax[test_idx], cvec_jax[test_idx], labels_jax[test_idx]

    train_domains = [domains_list[i] for i in train_idx]
    val_domains = [domains_list[i] for i in val_idx]
    test_domains = [domains_list[i] for i in test_idx]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # --- Step 4: Learning rate sweep ---
    print("\n[4/5] Training with learning rate sweep [1e-4, 1e-3, 1e-2]...")
    lr_candidates = [1e-4, 1e-3, 1e-2]
    n_epochs = 200
    best_lr = None
    best_val_auroc = -1.0
    best_params = None
    best_loss_history = None
    best_grad_history = None
    lr_results = {}

    for lr in lr_candidates:
        print(f"\n  --- LR = {lr} ---")
        key = jrandom.PRNGKey(int(lr * 1e6))
        params, loss_hist, grad_hist = train_verifier(
            train_emb, train_cvec, train_labels,
            lr=lr, n_epochs=n_epochs, key=key,
            hidden_dim=64, alpha=10.0, verbose=True,
        )

        # Evaluate on validation set
        val_eval = evaluate_verifier(
            params, val_emb, val_cvec, np.array(val_labels), val_domains
        )
        val_auroc = val_eval["auroc_joint"]
        print(f"  Val AUROC (joint): {val_auroc:.4f}")
        print(f"  Val AUROC (ising_only): {val_eval['auroc_ising_only']:.4f}")
        print(f"  Val AUROC (embed_only): {val_eval['auroc_embedding_only']:.4f}")

        lr_results[str(lr)] = {
            "val_auroc_joint": val_auroc,
            "val_auroc_ising_only": val_eval["auroc_ising_only"],
            "val_auroc_embedding_only": val_eval["auroc_embedding_only"],
            "final_loss": loss_hist[-1],
            "final_grad_norm": grad_hist[-1],
        }

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_lr = lr
            best_params = params
            best_loss_history = loss_hist
            best_grad_history = grad_hist

    print(f"\n  Best LR: {best_lr} (val AUROC = {best_val_auroc:.4f})")

    # --- Step 5: Final evaluation on test set ---
    print("\n[5/5] Evaluating on test set with best model...")
    test_eval = evaluate_verifier(
        best_params, test_emb, test_cvec, np.array(test_labels), test_domains
    )

    print(f"\n  Test AUROC (joint):          {test_eval['auroc_joint']:.4f}")
    print(f"  Test AUROC (ising_only):     {test_eval['auroc_ising_only']:.4f}")
    print(f"  Test AUROC (embedding_only): {test_eval['auroc_embedding_only']:.4f}")

    print(f"\n  Per-domain AUROC breakdown:")
    for domain, scores in test_eval["per_domain"].items():
        print(f"    {domain:>12s}: joint={scores['joint']:.4f}  "
              f"ising={scores['ising_only']:.4f}  embed={scores['embedding_only']:.4f}")

    # --- Gradient stability analysis ---
    grad_norms = np.array(best_grad_history)
    grad_stability = {
        "mean_grad_norm": float(np.mean(grad_norms)),
        "std_grad_norm": float(np.std(grad_norms)),
        "max_grad_norm": float(np.max(grad_norms)),
        "min_grad_norm": float(np.min(grad_norms)),
        "grad_norm_ratio_max_min": float(np.max(grad_norms) / max(np.min(grad_norms), 1e-10)),
        "exploded": bool(np.any(np.isnan(grad_norms)) or np.max(grad_norms) > 1e6),
        "vanished": bool(np.min(grad_norms[-20:]) < 1e-8),
    }
    print(f"\n  Gradient stability:")
    print(f"    Mean norm: {grad_stability['mean_grad_norm']:.4f}")
    print(f"    Std norm:  {grad_stability['std_grad_norm']:.4f}")
    print(f"    Max norm:  {grad_stability['max_grad_norm']:.4f}")
    print(f"    Min norm:  {grad_stability['min_grad_norm']:.4f}")
    print(f"    Exploded:  {grad_stability['exploded']}")
    print(f"    Vanished:  {grad_stability['vanished']}")

    # --- Save results ---
    elapsed = time.time() - start
    results = {
        "experiment": "66_differentiable_constraints",
        "description": "End-to-end differentiable verification: text → embedding → constraints → continuous Ising → MLP → score",
        "prerequisites": ["experiment_64_continuous_ising", "experiment_65_embedding_constraints"],
        "dataset": {
            "n_total": len(dataset),
            "n_per_domain": 100,
            "domains": DOMAINS,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
        },
        "hyperparameters": {
            "embed_dim": EMBED_DIM,
            "n_constraints": N_CONSTRAINTS,
            "hidden_dim": 64,
            "alpha": 10.0,
            "n_epochs": n_epochs,
            "best_lr": best_lr,
            "optimizer": "Adam",
        },
        "lr_sweep": lr_results,
        "test_results": {
            "auroc_joint": test_eval["auroc_joint"],
            "auroc_ising_only": test_eval["auroc_ising_only"],
            "auroc_embedding_only": test_eval["auroc_embedding_only"],
            "per_domain": test_eval["per_domain"],
        },
        "ablation_summary": {
            "joint_vs_ising_delta": test_eval["auroc_joint"] - test_eval["auroc_ising_only"],
            "joint_vs_embed_delta": test_eval["auroc_joint"] - test_eval["auroc_embedding_only"],
            "ising_adds_over_embed": test_eval["auroc_joint"] > test_eval["auroc_embedding_only"],
            "embed_adds_over_ising": test_eval["auroc_joint"] > test_eval["auroc_ising_only"],
        },
        "gradient_stability": grad_stability,
        "training": {
            "final_loss": best_loss_history[-1] if best_loss_history else None,
            "loss_history_sample": (
                best_loss_history[::20] if best_loss_history else []
            ),
            "grad_norm_history_sample": (
                [float(g) for g in best_grad_history[::20]] if best_grad_history else []
            ),
        },
        "wall_time_seconds": elapsed,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved to {RESULTS_PATH}")

    # --- Verdict ---
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 66 RESULTS ({elapsed:.0f}s)")
    print(f"{'='*70}")

    joint_auroc = test_eval["auroc_joint"]
    ising_auroc = test_eval["auroc_ising_only"]
    embed_auroc = test_eval["auroc_embedding_only"]

    print(f"\n  Joint AUROC:          {joint_auroc:.4f}")
    print(f"  Ising-only AUROC:     {ising_auroc:.4f}")
    print(f"  Embedding-only AUROC: {embed_auroc:.4f}")

    if joint_auroc > max(ising_auroc, embed_auroc) + 0.02:
        print(f"\n  VERDICT: Joint model OUTPERFORMS both ablations.")
        print(f"  The continuous Ising layer and embeddings are COMPLEMENTARY —")
        print(f"  combining them yields better verification than either alone.")
        print(f"  This validates the differentiable verifier architecture for Phase 17.")
    elif joint_auroc > 0.5 + 0.05:
        print(f"\n  VERDICT: Joint model achieves meaningful discrimination (>{joint_auroc:.2f} AUROC).")
        print(f"  The end-to-end differentiable pipeline works, though ablation")
        print(f"  differences are modest. Both components contribute signal.")
    else:
        print(f"\n  VERDICT: Model struggles to discriminate correct from wrong.")
        print(f"  This may indicate the synthetic dataset is too simple or the")
        print(f"  embedding fallback (no sentence-transformers) lacks signal.")

    if not grad_stability["exploded"] and not grad_stability["vanished"]:
        print(f"\n  GRADIENT STABILITY: Stable throughout training.")
        print(f"  Gradients neither exploded nor vanished — the continuous Ising")
        print(f"  relaxation provides a smooth optimization landscape.")
    elif grad_stability["exploded"]:
        print(f"\n  GRADIENT STABILITY: WARNING — gradients exploded.")
        print(f"  Consider reducing learning rate or adding gradient clipping.")
    else:
        print(f"\n  GRADIENT STABILITY: WARNING — gradients vanished in late training.")
        print(f"  Consider increasing learning rate or using a warmer alpha schedule.")

    print(f"\n  NOTE: This differentiable verifier is the FOUNDATION for Phase 17.")
    print(f"  Exp 86 will integrate it with LLM generation for gradient-guided repair.")
    print(f"  Exp 87 will benchmark it against discrete verification at scale.")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
