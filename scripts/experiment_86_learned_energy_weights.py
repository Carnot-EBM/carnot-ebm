#!/usr/bin/env python3
"""Experiment 86: Learned Energy Weights — Auto-tune constraint type weights via gradient descent.

**Researcher summary:**
    ComposedEnergy currently uses uniform weights (all 1.0) for each constraint
    term. This experiment learns per-type weights (arithmetic, type_check, bound,
    logical, etc.) via gradient descent on a binary cross-entropy loss: correct
    responses should produce low total energy, wrong responses high energy.
    Extends the differentiable pipeline from Exp 66.

**Detailed explanation for engineers:**
    The hypothesis is that different constraint types have different predictive
    power for verification accuracy. For example, arithmetic constraints may be
    highly diagnostic (if the math is wrong, the answer is wrong), while
    "reasonable_length" constraints may be weakly correlated with correctness.

    Instead of hand-tuning weights or leaving them uniform, we LEARN the optimal
    weights from labeled (correct/wrong) response pairs. The architecture:

    1. **LearnedComposedEnergy**: Wraps ComposedEnergy with trainable weights
       (one per constraint TYPE, not per instance). Uses softplus to keep weights
       positive. The forward pass is a weighted sum of per-type energies, fully
       differentiable via JAX.

    2. **Training**: Generate 500 synthetic QA pairs across 5 domains (reusing
       Exp 66's dataset generator). For each pair, extract constraints via
       AutoExtractor, compute per-type energies, and train weights with BCE loss
       (correct=low energy, wrong=high energy). Adam optimizer, 100 epochs.

    3. **Evaluation**: Compare uniform vs learned weights on held-out validation
       set using AUROC. Also train per-domain weights to see if different domains
       need different weight profiles. Bootstrap 95% CI on AUROC difference.

    4. **Key output**: The learned weight values themselves — which constraint
       types matter most for verification accuracy?

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_86_learned_energy_weights.py

REQ: REQ-EBT-001, REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-004
"""

from __future__ import annotations

import json
import os
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

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "experiment_86_results.json"
)

# All constraint types that AutoExtractor can produce, plus the 8 lightweight
# text heuristics from Exp 66. We group them into broad categories for
# per-type weight learning.
CONSTRAINT_TYPE_CATEGORIES = [
    "arithmetic",       # ArithmeticExtractor: "X + Y = Z" claims
    "type_check",       # CodeExtractor: parameter type annotations
    "return_type",      # CodeExtractor: return type annotations
    "bound",            # CodeExtractor: loop variable bounds
    "initialization",   # CodeExtractor: uninitialized variable checks
    "implication",      # LogicExtractor: "if P then Q" patterns
    "exclusion",        # LogicExtractor: "X but not Y" patterns
    "factual",          # NLExtractor: "X is Y" claims
    "quantity",         # NLExtractor: "there are N items" claims
    "heuristic",        # Lightweight text checks (from Exp 66)
]

# Lightweight text heuristic names (same as Exp 66).
HEURISTIC_NAMES = [
    "contains_number",
    "complete_sentence",
    "not_empty",
    "no_contradiction",
    "has_explanation",
    "reasonable_length",
    "no_repetition",
    "topic_relevant",
]

DOMAINS = ["arithmetic", "code", "logic", "factual", "scheduling"]

_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from it its this that these those and but or not no".split()
)


# ---------------------------------------------------------------------------
# 1. Multi-domain synthetic dataset (reused from Exp 66)
# ---------------------------------------------------------------------------


def _generate_multi_domain_dataset(
    n_per_domain: int = 100,
    seed: int = 86,
) -> list[dict]:
    """Generate 500 QA pairs across 5 domains with correct/wrong labels.

    **Detailed explanation for engineers:**
        Same structure as Exp 66's dataset generator but with a different
        random seed (86) so the specific numeric values differ. Each domain
        contributes n_per_domain pairs (half correct, half wrong), giving
        5 * 100 = 500 total pairs. The dataset is shuffled deterministically.

    Args:
        n_per_domain: Pairs per domain (half correct, half wrong).
        seed: Random seed for reproducibility.

    Returns:
        List of dicts: {domain, question, answer, is_correct}.
    """
    rng = np.random.default_rng(seed)
    dataset: list[dict] = []
    n_each = n_per_domain // 2

    # --- Arithmetic ---
    # Answers embed "A + B = C" patterns that ArithmeticExtractor can parse.
    for i in range(n_each):
        a = int(rng.integers(10, 500))
        b = int(rng.integers(1, 200))
        op = ["+", "-"][i % 2]  # Only +/- for ArithmeticExtractor pattern
        if op == "+":
            r = a + b
        else:
            r = a - b
        q = f"What is {a} {op} {b}?"
        dataset.append({
            "domain": "arithmetic",
            "question": q,
            "answer": f"We compute {a} {op} {b} = {r}. Because this is basic arithmetic.",
            "is_correct": True,
        })
        wrong_r = r + int(rng.integers(1, 50))
        dataset.append({
            "domain": "arithmetic",
            "question": q,
            "answer": f"We compute {a} {op} {b} = {wrong_r}. Because the calculation gives {wrong_r}.",
            "is_correct": False,
        })

    # --- Code ---
    code_tasks = [
        ("reverse a string", "def reverse_string(s: str) -> str:\n    return s[::-1]"),
        ("compute factorial", "def factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n-1)"),
        ("find maximum", "def find_max(lst: list) -> int:\n    return max(lst)"),
        ("check palindrome", "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]"),
        ("sum of list", "def sum_list(lst: list) -> int:\n    return sum(lst)"),
    ]
    for i in range(n_each):
        task_name, code = code_tasks[i % len(code_tasks)]
        q = f"Write a Python function to {task_name}."
        dataset.append({
            "domain": "code",
            "question": q,
            "answer": f"Here is the solution:\n```python\n{code}\n```\nThis works because it implements the required logic.",
            "is_correct": True,
        })
        wrong_answers = [
            "def func():\n    pass",
            "The answer is 42.",
            f"def {task_name.replace(' ', '_')}(x):\n    return None  # TODO",
        ]
        dataset.append({
            "domain": "code",
            "question": q,
            "answer": wrong_answers[i % len(wrong_answers)],
            "is_correct": False,
        })

    # --- Logic ---
    # Answers embed "if X, then Y" patterns that LogicExtractor can parse,
    # and "X is Y" patterns that NLExtractor can parse.
    logic_templates = [
        (
            "If it rains, the ground gets wet. It rained. Is the ground wet?",
            "If it rains, then the ground gets wet. It rained, so the ground is wet.",
            "If it rains, then the ground stays dry. The ground does not get wet.",
        ),
        (
            "All cats are mammals. Tom is a cat. Is Tom a mammal?",
            "All cats are mammals. If Tom is a cat, then Tom is a mammal.",
            "All cats are reptiles. Tom is not a mammal.",
        ),
        (
            "If A implies B, and A is true, then B must be?",
            "If A is true, then B is true. All implications are preserved.",
            "If A is true, then B is false. The implication does not hold.",
        ),
        (
            "If X > 5 and X < 10, and X = 7, is the condition satisfied?",
            "If X is 7, then X is greater than 5. All conditions are satisfied.",
            "If X is 7, then X is less than 5. The condition is not met.",
        ),
        (
            "All birds can fly. Penguins are birds. Can penguins fly?",
            "If penguins are birds, then penguins can fly. All birds are flyers.",
            "Penguins cannot fly. All birds are non-flyers.",
        ),
    ]
    for i in range(n_each):
        q, correct_a, wrong_a = logic_templates[i % len(logic_templates)]
        dataset.append({
            "domain": "logic",
            "question": q,
            "answer": correct_a,
            "is_correct": True,
        })
        dataset.append({
            "domain": "logic",
            "question": q,
            "answer": wrong_a,
            "is_correct": False,
        })

    # --- Factual ---
    # Answers embed "X is Y" and "X is the Y of Z" patterns for NLExtractor.
    factual_pairs = [
        ("What is the capital of France?",
         "Paris is the capital of France.",
         "Lyon is the capital of France."),
        ("What is the chemical symbol for water?",
         "H2O is the chemical symbol of water. There are 2 hydrogen atoms.",
         "CO2 is the chemical symbol of water. There are 2 oxygen atoms."),
        ("How many continents are there?",
         "There are 7 continents. This is a well-known fact.",
         "There are 5 continents. This is what I recall."),
        ("What planet is closest to the Sun?",
         "Mercury is the closest planet. Mercury is the first planet of the solar system.",
         "Venus is the closest planet. Venus is the first planet of the solar system."),
        ("What year did World War II end?",
         "World War II ended in 1945. 1945 is the year of victory.",
         "World War II ended in 1939. 1939 is the year of conclusion."),
    ]
    for i in range(n_each):
        q, correct_a, wrong_a = factual_pairs[i % len(factual_pairs)]
        dataset.append({
            "domain": "factual",
            "question": q,
            "answer": correct_a,
            "is_correct": True,
        })
        dataset.append({
            "domain": "factual",
            "question": q,
            "answer": wrong_a,
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
            "answer": (
                f"If Meeting A ends at {h2}:00, then Meeting B can start at {h3}:00. "
                f"There are 2 meetings. {h3} - {h2} = {h3 - h2} hours gap, so no conflict."
            ),
            "is_correct": True,
        })
        dataset.append({
            "domain": "scheduling",
            "question": q,
            "answer": (
                f"If Meeting A ends at {h2}:00, then Meeting B cannot start at {h3}:00. "
                f"There are 2 meetings. {h3} - {h2} = {h3 - h2 + int(rng.integers(5, 15))} overlap hours."
            ),
            "is_correct": False,
        })

    # Deterministic shuffle.
    indices = list(range(len(dataset)))
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(indices)
    dataset = [dataset[i] for i in indices]

    return dataset


# ---------------------------------------------------------------------------
# 2. Feature extraction: per-type energy vectors
# ---------------------------------------------------------------------------


def _check_heuristics(question: str, answer: str) -> list[bool]:
    """Evaluate 8 lightweight text heuristic constraints on an answer.

    **Detailed explanation for engineers:**
        Same heuristics as Exp 66. Each captures one structural aspect:
        contains numbers, complete sentence, non-empty, no contradiction,
        has explanation keywords, reasonable length, no repeated trigrams,
        topic relevance (word overlap with question).

    Args:
        question: The question text.
        answer: The candidate answer text.

    Returns:
        List of 8 booleans, one per HEURISTIC_NAMES entry.
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

    # 7. no_repetition (no repeated 3-word trigrams)
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    results.append(len(trigrams) == len(set(trigrams)))

    # 8. topic_relevant
    q_words = set(question.lower().split()) - _STOPWORDS
    a_words = set(answer.lower().split()) - _STOPWORDS
    results.append(len(q_words & a_words) > 0)

    return results


def _extract_constraint_type_counts(
    question: str,
    answer: str,
    extractor: AutoExtractor,
) -> dict[str, float]:
    """Extract per-constraint-type feature scores for a single QA pair.

    **Detailed explanation for engineers:**
        Runs AutoExtractor on the answer text to get structured constraints
        (arithmetic, type_check, bound, implication, etc.), then adds the
        8 lightweight heuristic checks. For each constraint TYPE, we compute
        a feature score in [0, 1] using TWO signals:

        1. **Satisfaction signal**: For types that set ``satisfied`` in
           metadata (arithmetic, bound, initialization, return_value_type),
           we use the fraction of satisfied constraints. This directly
           tells us if the constraint content is correct.

        2. **Presence signal**: For types that don't set ``satisfied``
           (implication, factual, quantity, exclusion, etc.), we use
           a normalized count: min(n_found / 3, 1.0). The insight is
           that correct answers tend to contain more well-structured
           extractable patterns than wrong answers (e.g., correct code
           has type annotations, correct logic has proper if-then chains).

        The "heuristic" type is the average of the 8 lightweight checks
        (always has a satisfaction signal).

    Args:
        question: Question text (for topic-relevance heuristic).
        answer: Response text to analyze.
        extractor: AutoExtractor instance.

    Returns:
        Dict mapping constraint type → score in [0, 1]. NaN for types
        with zero instances and no presence signal.
    """
    # Collect per-type constraints.
    type_satisfied: dict[str, list[bool]] = {t: [] for t in CONSTRAINT_TYPE_CATEGORIES}
    type_count: dict[str, int] = {t: 0 for t in CONSTRAINT_TYPE_CATEGORIES}

    # Extract structured constraints via AutoExtractor.
    try:
        constraints = extractor.extract(answer)
        for cr in constraints:
            ctype = cr.constraint_type
            # Map related types to our categories.
            if ctype == "return_value_type":
                ctype = "return_type"
            if ctype not in type_count:
                continue
            type_count[ctype] += 1
            satisfied = cr.metadata.get("satisfied")
            if satisfied is not None:
                type_satisfied[ctype].append(bool(satisfied))
    except Exception:
        pass

    # Add heuristic checks.
    heuristic_checks = _check_heuristics(question, answer)
    type_satisfied["heuristic"] = heuristic_checks
    type_count["heuristic"] = len(heuristic_checks)

    # Convert to feature scores.
    result: dict[str, float] = {}
    for ctype in CONSTRAINT_TYPE_CATEGORIES:
        sat_list = type_satisfied[ctype]
        count = type_count[ctype]

        if sat_list:
            # Use satisfaction ratio when we have satisfaction data.
            result[ctype] = sum(1.0 for c in sat_list if c) / len(sat_list)
        elif count > 0:
            # Use presence signal: more constraints → higher satisfaction.
            # Maps to [0.5, 1.0] range so that ANY presence is above the
            # neutral default of 0.5. This ensures that extracting more
            # well-structured constraints (type annotations, if-then chains)
            # always indicates higher quality than having none.
            result[ctype] = 0.5 + 0.5 * min(count / 3.0, 1.0)
        else:
            result[ctype] = float("nan")
    return result


def _build_feature_matrix(
    dataset: list[dict],
    extractor: AutoExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-type satisfaction feature matrix and label vector for the full dataset.

    **Detailed explanation for engineers:**
        For each QA pair, extracts per-constraint-type satisfaction ratios.
        NaN values (types with no instances for that pair) are replaced with
        0.5 (neutral). Returns a float32 matrix of shape (n_samples, n_types)
        and a float32 label vector of shape (n_samples,) where 1.0 = correct
        and 0.0 = wrong.

    Args:
        dataset: List of QA pair dicts from _generate_multi_domain_dataset.
        extractor: AutoExtractor instance.

    Returns:
        (features, labels) — both as numpy float32 arrays.
    """
    n = len(dataset)
    n_types = len(CONSTRAINT_TYPE_CATEGORIES)
    features = np.full((n, n_types), 0.5, dtype=np.float32)
    labels = np.zeros(n, dtype=np.float32)

    for i, item in enumerate(dataset):
        ratios = _extract_constraint_type_counts(
            item["question"], item["answer"], extractor
        )
        for j, ctype in enumerate(CONSTRAINT_TYPE_CATEGORIES):
            val = ratios.get(ctype, float("nan"))
            if not np.isnan(val):
                features[i, j] = val
        labels[i] = 1.0 if item["is_correct"] else 0.0

    return features, labels


# ---------------------------------------------------------------------------
# 3. LearnedComposedEnergy — trainable per-type weights
# ---------------------------------------------------------------------------


class LearnedWeightsParams:
    """JAX pytree holding trainable per-type weights for energy composition.

    **Detailed explanation for engineers:**
        Stores raw (unconstrained) weight parameters. The actual weights used
        in the energy computation are softplus(raw_weights) to ensure positivity.
        Registered as a JAX pytree so jax.grad can differentiate through the
        entire forward pass including weight application.

        The raw weights are initialized to zeros, which maps to softplus(0) ≈ 0.693
        (close to uniform weight of 1.0 but not exactly, since we normalize later).

    Attributes:
        raw_weights: Unconstrained weight parameters, shape (n_types,).
    """

    def __init__(self, raw_weights: jax.Array) -> None:
        self.raw_weights = raw_weights

    def tree_flatten(self):
        return (self.raw_weights,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])

    def get_weights(self) -> jax.Array:
        """Return positive weights via softplus transform.

        **Detailed explanation for engineers:**
            softplus(x) = log(1 + exp(x)) is a smooth approximation to ReLU
            that is always positive and differentiable everywhere. This ensures
            the optimization can freely explore weight space without needing
            constrained optimization.

        Returns:
            Positive weight vector, shape (n_types,).
        """
        return jax.nn.softplus(self.raw_weights)


jax.tree_util.register_pytree_node(
    LearnedWeightsParams,
    lambda p: p.tree_flatten(),
    lambda aux, children: LearnedWeightsParams.tree_unflatten(aux, children),
)


def init_weights_params(n_types: int, key: jax.Array) -> LearnedWeightsParams:
    """Initialize weight parameters to small random values near zero.

    **Detailed explanation for engineers:**
        Starting near zero means softplus(~0) ≈ 0.693 for all types — roughly
        uniform. The small random perturbation breaks symmetry so gradient
        descent can differentiate the types from the start.

    Args:
        n_types: Number of constraint type categories.
        key: JAX PRNG key.

    Returns:
        Initialized LearnedWeightsParams.
    """
    raw = jrandom.normal(key, (n_types,)) * 0.1
    return LearnedWeightsParams(raw)


def weighted_energy(
    params: LearnedWeightsParams,
    satisfaction_vec: jax.Array,
) -> jax.Array:
    """Compute weighted energy from per-type satisfaction scores.

    **Detailed explanation for engineers:**
        Energy = sum_i w_i * (1 - s_i), where w_i is the learned weight for
        type i and s_i is the satisfaction ratio for type i. This means:
        - Fully satisfied (s_i = 1) contributes 0 energy.
        - Fully violated (s_i = 0) contributes w_i energy.
        - The weights control how much each type MATTERS for the total energy.

        The weights are obtained via softplus to ensure positivity. The total
        energy is a scalar that should be LOW for correct responses and HIGH
        for wrong responses.

    Args:
        params: LearnedWeightsParams with trainable weights.
        satisfaction_vec: Per-type satisfaction ratios, shape (n_types,).

    Returns:
        Scalar total weighted energy (non-negative).
    """
    weights = params.get_weights()
    violations = 1.0 - satisfaction_vec
    return jnp.dot(weights, violations)


# ---------------------------------------------------------------------------
# 4. Loss function and training
# ---------------------------------------------------------------------------


def bce_loss_single(
    params: LearnedWeightsParams,
    satisfaction_vec: jax.Array,
    label: jax.Array,
    margin: float = 2.0,
) -> jax.Array:
    """Binary cross-entropy loss for a single sample.

    **Detailed explanation for engineers:**
        Converts the weighted energy into a probability via sigmoid:
            p(correct) = sigmoid(margin - energy)
        So low energy → high p(correct), high energy → low p(correct).
        The margin parameter controls the decision boundary: with margin=2.0,
        an energy of 2.0 maps to sigmoid(0) = 0.5 (uncertain).

        BCE = -[y * log(p) + (1-y) * log(1-p)]
        where y is the label (1.0 = correct, 0.0 = wrong).

        Clipping prevents log(0) NaN.

    Args:
        params: LearnedWeightsParams.
        satisfaction_vec: Per-type satisfaction, shape (n_types,).
        label: Binary label (1.0 = correct, 0.0 = wrong).
        margin: Energy margin for sigmoid decision boundary.

    Returns:
        Scalar BCE loss (lower is better).
    """
    energy = weighted_energy(params, satisfaction_vec)
    # sigmoid(margin - energy): low energy → high probability of correct
    prob = jax.nn.sigmoid(margin - energy)
    prob = jnp.clip(prob, 1e-7, 1.0 - 1e-7)
    return -(label * jnp.log(prob) + (1.0 - label) * jnp.log(1.0 - prob))


def batch_loss(
    params: LearnedWeightsParams,
    features: jax.Array,
    labels: jax.Array,
    margin: float = 2.0,
) -> jax.Array:
    """Mean BCE loss over a batch.

    **Detailed explanation for engineers:**
        Uses jax.vmap to vectorize the single-sample loss across the batch.
        This is more efficient than a Python loop and enables JAX to compile
        the entire batch computation into a single XLA kernel.

    Args:
        params: LearnedWeightsParams.
        features: Batch of satisfaction vectors, shape (batch, n_types).
        labels: Batch of labels, shape (batch,).
        margin: Energy margin.

    Returns:
        Scalar mean loss over the batch.
    """
    losses = jax.vmap(
        lambda feat, lab: bce_loss_single(params, feat, lab, margin)
    )(features, labels)
    return jnp.mean(losses)


def train_weights(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_val: np.ndarray,
    labels_val: np.ndarray,
    n_epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
    margin: float = 2.0,
) -> tuple[LearnedWeightsParams, list[dict]]:
    """Train per-type weights via Adam on BCE loss.

    **Detailed explanation for engineers:**
        Standard gradient descent training loop using JAX. We implement Adam
        manually (rather than importing optax) to keep dependencies minimal.
        Adam's adaptive learning rates help because different constraint types
        may have very different gradient magnitudes.

        The training history records loss and validation AUROC every 10 epochs
        so we can check for overfitting. Early stopping is NOT used because
        with only ~10 parameters (one weight per type), overfitting is unlikely.

    Args:
        features_train: Training satisfaction vectors, shape (n_train, n_types).
        labels_train: Training labels, shape (n_train,).
        features_val: Validation satisfaction vectors, shape (n_val, n_types).
        labels_val: Validation labels, shape (n_val,).
        n_epochs: Number of training epochs.
        lr: Learning rate for Adam.
        seed: Random seed for parameter initialization.
        margin: Energy margin for BCE sigmoid.

    Returns:
        (trained_params, history) where history is a list of dicts with
        epoch, train_loss, val_loss, val_auroc.
    """
    key = jrandom.PRNGKey(seed)
    n_types = features_train.shape[1]
    params = init_weights_params(n_types, key)

    # Convert to JAX arrays.
    X_train = jnp.array(features_train)
    y_train = jnp.array(labels_train)
    X_val = jnp.array(features_val)
    y_val = jnp.array(labels_val)

    # Adam optimizer state.
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    grad_fn = jax.grad(lambda p: batch_loss(p, X_train, y_train, margin))
    history: list[dict] = []

    for epoch in range(1, n_epochs + 1):
        grads = grad_fn(params)

        # Adam update.
        m = jax.tree.map(lambda mi, gi: beta1 * mi + (1 - beta1) * gi, m, grads)
        v = jax.tree.map(lambda vi, gi: beta2 * vi + (1 - beta2) * gi ** 2, v, grads)

        m_hat = jax.tree.map(lambda mi: mi / (1 - beta1 ** epoch), m)
        v_hat = jax.tree.map(lambda vi: vi / (1 - beta2 ** epoch), v)

        params = jax.tree.map(
            lambda pi, mhi, vhi: pi - lr * mhi / (jnp.sqrt(vhi) + eps),
            params, m_hat, v_hat,
        )

        if epoch % 10 == 0 or epoch == 1:
            train_loss = float(batch_loss(params, X_train, y_train, margin))
            val_loss = float(batch_loss(params, X_val, y_val, margin))
            val_auroc = _compute_auroc(params, X_val, y_val, margin)
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auroc": val_auroc,
            })
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_auroc={val_auroc:.4f}")

    return params, history


# ---------------------------------------------------------------------------
# 5. Evaluation: AUROC and bootstrap CI
# ---------------------------------------------------------------------------


def _compute_auroc(
    params: LearnedWeightsParams,
    features: jax.Array,
    labels: jax.Array,
    margin: float = 2.0,
) -> float:
    """Compute AUROC (area under ROC curve) for the learned energy scorer.

    **Detailed explanation for engineers:**
        AUROC measures how well the energy score discriminates correct from
        wrong responses. An AUROC of 1.0 means perfect separation; 0.5 means
        no better than random. We compute it using the rank-based formula
        (equivalent to Wilcoxon-Mann-Whitney U statistic / (n_pos * n_neg)).

        The score for each sample is sigmoid(margin - energy), so higher
        score = more likely correct. We sort samples by score and count
        how often correct samples are ranked above wrong ones.

    Args:
        params: Trained LearnedWeightsParams.
        features: Satisfaction vectors, shape (n, n_types).
        labels: Binary labels, shape (n,).
        margin: Energy margin for sigmoid.

    Returns:
        AUROC as a float in [0, 1].
    """
    energies = jax.vmap(lambda f: weighted_energy(params, f))(features)
    scores = jax.nn.sigmoid(margin - energies)
    scores_np = np.array(scores)
    labels_np = np.array(labels)

    # Rank-based AUROC computation.
    pos_scores = scores_np[labels_np == 1.0]
    neg_scores = scores_np[labels_np == 0.0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5

    # Count how often a positive sample has a higher score than a negative one.
    n_correct = 0
    n_total = len(pos_scores) * len(neg_scores)
    for ps in pos_scores:
        n_correct += np.sum(ps > neg_scores)
        n_correct += 0.5 * np.sum(ps == neg_scores)

    return float(n_correct / n_total)


def _uniform_auroc(
    features: np.ndarray,
    labels: np.ndarray,
    margin: float = 2.0,
) -> float:
    """Compute AUROC with uniform weights (all 1.0) as baseline.

    **Detailed explanation for engineers:**
        Creates a LearnedWeightsParams with raw_weights set so that
        softplus(raw) = 1.0 for all types. softplus(x) = 1.0 when
        x = log(e - 1) ≈ 0.5413. This gives us the baseline AUROC
        that uniform weighting achieves.

    Args:
        features: Satisfaction vectors, shape (n, n_types).
        labels: Labels, shape (n,).
        margin: Energy margin.

    Returns:
        AUROC with uniform weights.
    """
    n_types = features.shape[1]
    # softplus(x) = 1.0 → x = log(e - 1) ≈ 0.5413
    raw_val = float(jnp.log(jnp.exp(jnp.array(1.0)) - 1.0))
    uniform_params = LearnedWeightsParams(jnp.full(n_types, raw_val))
    return _compute_auroc(
        uniform_params,
        jnp.array(features),
        jnp.array(labels),
        margin,
    )


def _bootstrap_auroc_ci(
    params: LearnedWeightsParams,
    features: np.ndarray,
    labels: np.ndarray,
    uniform_auroc: float,
    n_bootstrap: int = 1000,
    margin: float = 2.0,
    seed: int = 99,
) -> dict:
    """Bootstrap 95% confidence interval on AUROC improvement over uniform.

    **Detailed explanation for engineers:**
        Resamples the validation set with replacement n_bootstrap times.
        For each bootstrap sample, computes both learned and uniform AUROC,
        then takes the difference (learned - uniform). The 2.5th and 97.5th
        percentiles of the difference distribution give the 95% CI.

        If the CI excludes 0.0, the improvement is statistically significant
        at the 5% level.

    Args:
        params: Trained weights.
        features: Validation features.
        labels: Validation labels.
        uniform_auroc: Pre-computed uniform AUROC (for reference).
        n_bootstrap: Number of bootstrap resamples.
        margin: Energy margin.
        seed: Random seed.

    Returns:
        Dict with keys: mean_diff, ci_lower, ci_upper, significant.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    n_types = features.shape[1]
    raw_val = float(jnp.log(jnp.exp(jnp.array(1.0)) - 1.0))
    uniform_p = LearnedWeightsParams(jnp.full(n_types, raw_val))

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_feats = jnp.array(features[idx])
        boot_labels = jnp.array(labels[idx])

        learned_auc = _compute_auroc(params, boot_feats, boot_labels, margin)
        uniform_auc = _compute_auroc(uniform_p, boot_feats, boot_labels, margin)
        diffs.append(learned_auc - uniform_auc)

    diffs_arr = np.array(diffs)
    return {
        "mean_diff": float(np.mean(diffs_arr)),
        "ci_lower": float(np.percentile(diffs_arr, 2.5)),
        "ci_upper": float(np.percentile(diffs_arr, 97.5)),
        "significant": bool(
            np.percentile(diffs_arr, 2.5) > 0.0
            or np.percentile(diffs_arr, 97.5) < 0.0
        ),
    }


# ---------------------------------------------------------------------------
# 6. Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 86: Learned Energy Weights.

    **Detailed explanation for engineers:**
        Orchestrates the full experiment:
        1. Generate 500 synthetic QA pairs across 5 domains.
        2. Extract per-type constraint satisfaction features.
        3. Split 80/20 train/val.
        4. Train global learned weights (all domains pooled).
        5. Train per-domain learned weights.
        6. Evaluate: compare uniform vs learned AUROC, bootstrap CI.
        7. Print learned weight values.
        8. Save results to JSON.
    """
    start_time = time.time()
    print("=" * 70)
    print("Experiment 86: Learned Energy Weights")
    print("=" * 70)

    # Step 1: Generate dataset.
    print("\n--- Step 1: Generating multi-domain dataset (500 QA pairs) ---")
    dataset = _generate_multi_domain_dataset(n_per_domain=100, seed=86)
    print(f"  Generated {len(dataset)} QA pairs across {len(DOMAINS)} domains.")
    for domain in DOMAINS:
        count = sum(1 for d in dataset if d["domain"] == domain)
        correct = sum(1 for d in dataset if d["domain"] == domain and d["is_correct"])
        print(f"  {domain}: {count} pairs ({correct} correct, {count - correct} wrong)")

    # Step 2: Extract features.
    print("\n--- Step 2: Extracting per-type constraint features ---")
    extractor = AutoExtractor()
    features, labels = _build_feature_matrix(dataset, extractor)
    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Labels: {int(labels.sum())} correct, {int(len(labels) - labels.sum())} wrong")

    # Show which constraint types have non-trivial coverage.
    for j, ctype in enumerate(CONSTRAINT_TYPE_CATEGORIES):
        non_neutral = np.sum(features[:, j] != 0.5)
        mean_val = np.mean(features[:, j])
        print(f"  {ctype:20s}: {non_neutral:3d} non-neutral samples, mean={mean_val:.3f}")

    # Step 3: Train/val split.
    print("\n--- Step 3: Train/val split (80/20) ---")
    n = len(dataset)
    n_train = int(0.8 * n)
    rng_split = np.random.default_rng(86)
    perm = rng_split.permutation(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]
    print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

    # Step 4: Baseline — uniform weights AUROC.
    print("\n--- Step 4: Baseline — uniform weights ---")
    uniform_auc = _uniform_auroc(X_val, y_val)
    print(f"  Uniform weights AUROC: {uniform_auc:.4f}")

    # Step 5: Train global learned weights.
    print("\n--- Step 5: Training global learned weights (200 epochs) ---")
    learned_params, train_history = train_weights(
        X_train, y_train, X_val, y_val,
        n_epochs=200, lr=5e-3, seed=42,
    )

    learned_auc = _compute_auroc(
        learned_params,
        jnp.array(X_val),
        jnp.array(y_val),
    )
    print(f"\n  Learned weights AUROC: {learned_auc:.4f}")
    print(f"  Improvement over uniform: {learned_auc - uniform_auc:+.4f}")

    # Print learned weight values.
    print("\n  Learned weight values (softplus-transformed):")
    learned_weights = np.array(learned_params.get_weights())
    weight_ranking = sorted(
        zip(CONSTRAINT_TYPE_CATEGORIES, learned_weights),
        key=lambda x: -x[1],
    )
    for ctype, w in weight_ranking:
        print(f"    {ctype:20s}: {w:.4f}")

    # Step 6: Bootstrap CI on AUROC difference.
    print("\n--- Step 6: Bootstrap 95% CI on AUROC improvement ---")
    ci_result = _bootstrap_auroc_ci(
        learned_params, X_val, y_val, uniform_auc,
        n_bootstrap=1000, seed=99,
    )
    print(f"  Mean AUROC diff (learned - uniform): {ci_result['mean_diff']:+.4f}")
    print(f"  95% CI: [{ci_result['ci_lower']:+.4f}, {ci_result['ci_upper']:+.4f}]")
    print(f"  Statistically significant: {ci_result['significant']}")

    # Step 7: Per-domain learned weights.
    print("\n--- Step 7: Per-domain learned weights ---")
    domain_results = {}
    domains_in_data = [d["domain"] for d in dataset]

    for domain in DOMAINS:
        print(f"\n  === Domain: {domain} ===")
        # Filter to this domain.
        domain_mask_train = np.array([
            domains_in_data[i] == domain for i in train_idx
        ])
        domain_mask_val = np.array([
            domains_in_data[i] == domain for i in val_idx
        ])

        if domain_mask_train.sum() < 10 or domain_mask_val.sum() < 5:
            print(f"    Skipping (too few samples: train={domain_mask_train.sum()}, val={domain_mask_val.sum()})")
            domain_results[domain] = {"skipped": True}
            continue

        X_d_train = X_train[domain_mask_train]
        y_d_train = y_train[domain_mask_train]
        X_d_val = X_val[domain_mask_val]
        y_d_val = y_val[domain_mask_val]

        # Uniform baseline for this domain.
        d_uniform_auc = _uniform_auroc(X_d_val, y_d_val)

        # Train domain-specific weights.
        d_params, d_history = train_weights(
            X_d_train, y_d_train, X_d_val, y_d_val,
            n_epochs=200, lr=5e-3, seed=42,
        )

        d_learned_auc = _compute_auroc(
            d_params,
            jnp.array(X_d_val),
            jnp.array(y_d_val),
        )

        d_weights = {
            ctype: float(w)
            for ctype, w in zip(
                CONSTRAINT_TYPE_CATEGORIES, np.array(d_params.get_weights())
            )
        }

        print(f"    Uniform AUROC: {d_uniform_auc:.4f}")
        print(f"    Learned AUROC: {d_learned_auc:.4f}")
        print(f"    Improvement: {d_learned_auc - d_uniform_auc:+.4f}")
        print(f"    Top weights: ", end="")
        top_3 = sorted(d_weights.items(), key=lambda x: -x[1])[:3]
        print(", ".join(f"{k}={v:.3f}" for k, v in top_3))

        domain_results[domain] = {
            "uniform_auroc": d_uniform_auc,
            "learned_auroc": d_learned_auc,
            "improvement": d_learned_auc - d_uniform_auc,
            "weights": d_weights,
            "n_train": int(domain_mask_train.sum()),
            "n_val": int(domain_mask_val.sum()),
        }

    # Step 8: Save results.
    elapsed = time.time() - start_time
    print(f"\n--- Step 8: Saving results (elapsed: {elapsed:.1f}s) ---")

    results = {
        "experiment": "86_learned_energy_weights",
        "description": (
            "Auto-tune per-constraint-type weights for ComposedEnergy via "
            "gradient descent on BCE loss. Compares uniform vs learned weights "
            "on verification AUROC."
        ),
        "target_models": ["Qwen3.5-0.8B", "google/gemma-4-E4B-it"],
        "dataset": {
            "n_samples": len(dataset),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "domains": DOMAINS,
            "n_per_domain": 100,
        },
        "constraint_types": CONSTRAINT_TYPE_CATEGORIES,
        "global_results": {
            "uniform_auroc": uniform_auc,
            "learned_auroc": learned_auc,
            "improvement": learned_auc - uniform_auc,
            "learned_weights": {
                ctype: float(w)
                for ctype, w in zip(
                    CONSTRAINT_TYPE_CATEGORIES,
                    np.array(learned_params.get_weights()),
                )
            },
            "bootstrap_ci": ci_result,
            "training_history": train_history,
        },
        "per_domain_results": domain_results,
        "elapsed_seconds": elapsed,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {RESULTS_PATH}")

    # Final summary.
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Global AUROC: uniform={uniform_auc:.4f}, learned={learned_auc:.4f} "
          f"(Δ={learned_auc - uniform_auc:+.4f})")
    print(f"  Bootstrap CI: [{ci_result['ci_lower']:+.4f}, {ci_result['ci_upper']:+.4f}]")
    print(f"  Significant: {ci_result['significant']}")
    print(f"\n  Most important constraint types (by learned weight):")
    for ctype, w in weight_ranking[:5]:
        print(f"    {ctype:20s}: {w:.4f}")
    print(f"\n  Per-domain improvements:")
    for domain in DOMAINS:
        dr = domain_results.get(domain, {})
        if dr.get("skipped"):
            print(f"    {domain}: skipped (too few samples)")
        else:
            print(f"    {domain}: Δ={dr.get('improvement', 0):+.4f}")
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
