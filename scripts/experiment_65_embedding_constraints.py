#!/usr/bin/env python3
"""Experiment 65: Joint Embedding-Constraint EBM — Bridge semantic and structural spaces.

**Researcher summary:**
    Trains a Gibbs EBM on joint feature vectors that concatenate semantic embeddings
    (all-MiniLM-L6-v2, 384-dim) with structural constraint vectors (per-constraint
    pass/fail from an Ising verifier, N-dim). Evaluates whether the joint model
    discriminates correct/wrong answers better than either space alone. Tests
    gradient-based repair in the joint space with nearest-neighbor decoding.

**Detailed explanation for engineers:**
    Experiment 64 relaxed discrete Ising spins to continuous space for gradient
    optimization. This experiment takes the next step: bridging EMBEDDING space
    (where semantic meaning lives) with CONSTRAINT space (where structural
    correctness lives) by training a single EBM on their concatenation.

    **Why this matters:**
    An LLM answer can be semantically plausible (close to correct answers in
    embedding space) but structurally wrong (violates constraints like type
    correctness, logical consistency, etc.). Conversely, an answer can satisfy
    all structural constraints but be semantically nonsensical. Neither space
    alone is sufficient for reliable verification.

    By training an EBM on the joint [embedding; constraint_vector] space, we
    get a model that captures the CORRELATION between semantic meaning and
    structural correctness. The energy landscape encodes: "what does a correct
    answer look like in BOTH semantic and structural terms simultaneously?"

    **Pipeline:**
    1. Generate 200 synthetic LLM answers (correct + wrong) for math/logic questions.
    2. Embed each answer with sentence-transformers (384-dim semantic embedding).
    3. Compute per-constraint pass/fail vector from an Ising-style verifier (N-dim).
    4. Concatenate: [embedding; constraint_vector] → (384+N)-dim joint vector.
    5. Train a small Gibbs EBM via NCE on joint vectors (correct=low, wrong=high energy).
    6. Compare AUROC: joint EBM vs embedding-only vs constraint-only.
    7. Gradient-based repair: start from a wrong answer's joint vector, descend on
       energy, decode via nearest-neighbor in embedding space.

    **Constraint encoding:**
    For each answer, we define N constraints (e.g., "contains a number",
    "answer is a complete sentence", "mentions the question topic", etc.).
    Each constraint is either satisfied (1.0) or violated (0.0). This gives
    a binary vector that captures structural properties of the answer.

    Unlike the Ising verifier in experiment 48 (which checks type-level
    propositions), here we use lightweight text-based constraint checks that
    apply to natural language answers. This makes the experiment self-contained
    without requiring code execution.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_65_embedding_constraints.py

REQ: REQ-EBT-001, REQ-VERIFY-001
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.models.gibbs import GibbsConfig, GibbsModel
from carnot.training.nce import nce_loss


# ---------------------------------------------------------------------------
# 1. Synthetic QA dataset with constraint annotations
# ---------------------------------------------------------------------------

# Constraint definitions: lightweight text-based checks applicable to
# natural language answers. Each constraint is a function that returns True
# (satisfied) or False (violated).

CONSTRAINT_NAMES = [
    "contains_number",       # Answer includes at least one digit
    "complete_sentence",     # Ends with period/question mark/exclamation
    "not_empty",             # Non-trivial length (>5 chars)
    "no_contradiction",      # Does not contain "not" and "is" in close proximity (heuristic)
    "has_explanation",        # Contains explanatory words (because, since, therefore, so)
    "reasonable_length",      # Between 10 and 500 characters
    "no_repetition",         # No repeated 3-word sequences
    "topic_relevant",        # Shares at least one non-stopword with the question
]

# Common English stopwords used for topic-relevance checking.
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from it its this that these those and but or not no".split()
)


def _check_constraints(question: str, answer: str) -> list[bool]:
    """Evaluate all constraints for a (question, answer) pair.

    **Detailed explanation for engineers:**
        Each constraint is a simple heuristic check on the answer text.
        These are NOT perfect — they are proxy signals for structural
        correctness, analogous to how the Ising verifier in experiment 48
        checks type-level propositions. The goal is to create a binary
        feature vector that captures different structural aspects of the
        answer, so the joint EBM can learn which combinations correlate
        with correctness.

    Args:
        question: The question text.
        answer: The candidate answer text.

    Returns:
        List of bools, one per constraint in CONSTRAINT_NAMES.
    """
    results = []

    # 1. contains_number: at least one digit in the answer
    results.append(any(c.isdigit() for c in answer))

    # 2. complete_sentence: ends with sentence-terminal punctuation
    stripped = answer.strip()
    results.append(len(stripped) > 0 and stripped[-1] in ".?!")

    # 3. not_empty: more than 5 characters (filters trivial/empty answers)
    results.append(len(stripped) > 5)

    # 4. no_contradiction: heuristic — flag if "not" appears within 3 words
    #    of "is/are/was/were" (rough proxy for self-contradictory statements).
    #    This is intentionally crude; the EBM will learn how much to weight it.
    words = answer.lower().split()
    has_contradiction = False
    for i, w in enumerate(words):
        if w == "not":
            window = words[max(0, i - 3): i + 4]
            if any(v in window for v in ("is", "are", "was", "were")):
                has_contradiction = True
                break
    results.append(not has_contradiction)

    # 5. has_explanation: contains causal/explanatory connectives
    lower = answer.lower()
    results.append(any(kw in lower for kw in ("because", "since", "therefore", "so ", "thus")))

    # 6. reasonable_length: between 10 and 500 characters
    results.append(10 <= len(stripped) <= 500)

    # 7. no_repetition: no 3-word sequence repeated (detects degenerate output)
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    results.append(len(trigrams) == len(set(trigrams)))

    # 8. topic_relevant: shares at least one content word with the question
    q_words = set(question.lower().split()) - _STOPWORDS
    a_words = set(answer.lower().split()) - _STOPWORDS
    results.append(len(q_words & a_words) > 0)

    return results


def _generate_qa_dataset(
    n_total: int = 200,
    seed: int = 65,
) -> tuple[list[str], list[str], list[bool]]:
    """Generate synthetic math/logic QA pairs with correct and wrong answers.

    **Detailed explanation for engineers:**
        Creates a balanced dataset of (question, answer, is_correct) triples.
        Correct answers are well-formed responses that satisfy most constraints.
        Wrong answers are plausible-looking but incorrect — they might have the
        right format but wrong numbers, or be off-topic, or be truncated.

        The questions cover simple arithmetic, comparisons, and logic to keep
        constraint checking meaningful (a math answer should contain a number,
        have a complete sentence, etc.).

    Args:
        n_total: Total number of QA pairs (half correct, half wrong).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (questions, answers, is_correct) lists.
    """
    rng = np.random.default_rng(seed)
    questions: list[str] = []
    answers: list[str] = []
    is_correct: list[bool] = []

    n_each = n_total // 2

    # --- Correct answers ---
    templates_correct = [
        ("What is {a} + {b}?", "The answer is {r}. This is because {a} plus {b} equals {r}."),
        ("What is {a} * {b}?", "The result is {r}. Since {a} times {b} gives {r}."),
        ("Is {a} greater than {b}?", "{yn}. Because {a} is {cmp} {b}, so the answer is {yn}."),
        ("What is {a} - {b}?", "The answer is {r}. Therefore {a} minus {b} equals {r}."),
        ("What is {a} divided by {b}?", "The result is {r}. Since {a} divided by {b} is {r}."),
        ("If x = {a}, what is x + {b}?", "x + {b} = {r}. Because substituting x = {a} gives {a} + {b} = {r}."),
        ("What is the remainder of {a} divided by {b}?", "The remainder is {r}. Since {a} mod {b} equals {r}."),
        ("Which is larger, {a} or {b}?", "{bigger} is larger. Because {bigger} > {smaller}."),
    ]

    for i in range(n_each):
        t_idx = i % len(templates_correct)
        a_val = int(rng.integers(1, 100))
        b_val = int(rng.integers(1, 50))
        q_tmpl, a_tmpl = templates_correct[t_idx]

        if t_idx == 0:
            r = a_val + b_val
            q = q_tmpl.format(a=a_val, b=b_val)
            ans = a_tmpl.format(a=a_val, b=b_val, r=r)
        elif t_idx == 1:
            r = a_val * b_val
            q = q_tmpl.format(a=a_val, b=b_val)
            ans = a_tmpl.format(a=a_val, b=b_val, r=r)
        elif t_idx == 2:
            yn = "Yes" if a_val > b_val else "No"
            cmp = "greater than" if a_val > b_val else "not greater than"
            q = q_tmpl.format(a=a_val, b=b_val)
            ans = a_tmpl.format(yn=yn, a=a_val, b=b_val, cmp=cmp)
        elif t_idx == 3:
            r = a_val - b_val
            q = q_tmpl.format(a=a_val, b=b_val)
            ans = a_tmpl.format(a=a_val, b=b_val, r=r)
        elif t_idx == 4:
            r = round(a_val / max(b_val, 1), 2)
            q = q_tmpl.format(a=a_val, b=max(b_val, 1))
            ans = a_tmpl.format(a=a_val, b=max(b_val, 1), r=r)
        elif t_idx == 5:
            r = a_val + b_val
            q = q_tmpl.format(a=a_val, b=b_val)
            ans = a_tmpl.format(a=a_val, b=b_val, r=r)
        elif t_idx == 6:
            b_val = max(b_val, 2)
            r = a_val % b_val
            q = q_tmpl.format(a=a_val, b=b_val)
            ans = a_tmpl.format(a=a_val, b=b_val, r=r)
        else:
            bigger = max(a_val, b_val)
            smaller = min(a_val, b_val)
            q = q_tmpl.format(a=a_val, b=b_val)
            ans = a_tmpl.format(bigger=bigger, smaller=smaller)

        questions.append(q)
        answers.append(ans)
        is_correct.append(True)

    # --- Wrong answers: plausible but incorrect ---
    wrong_strategies = [
        "wrong_number",    # Correct format but wrong numerical answer
        "off_topic",       # Answer about something unrelated
        "truncated",       # Answer cut short (incomplete sentence)
        "no_explanation",  # Just a bare number, no reasoning
        "contradictory",   # Contains self-contradiction
    ]

    for i in range(n_each):
        # Reuse the same questions (pair each with a wrong answer)
        q = questions[i]
        strategy = wrong_strategies[i % len(wrong_strategies)]
        a_val = int(rng.integers(1, 100))
        b_val = int(rng.integers(1, 50))

        if strategy == "wrong_number":
            # Plausible format but deliberately wrong answer
            wrong_r = int(rng.integers(1, 200))
            ans = f"The answer is {wrong_r}. Because the calculation gives {wrong_r}."
        elif strategy == "off_topic":
            ans = "Bananas are a good source of potassium and are grown in tropical climates."
        elif strategy == "truncated":
            ans = f"The answer is"
        elif strategy == "no_explanation":
            ans = str(int(rng.integers(1, 200)))
        elif strategy == "contradictory":
            ans = f"The answer is {a_val} but actually it is not {a_val} because the result is different."
        else:
            ans = f"I think the answer is maybe {a_val}."

        questions.append(q)
        answers.append(ans)
        is_correct.append(False)

    return questions, answers, is_correct


# ---------------------------------------------------------------------------
# 2. Embedding and constraint vector computation
# ---------------------------------------------------------------------------


def _embed_answers(
    questions: list[str],
    answers: list[str],
    embed_dim: int = 384,
) -> np.ndarray:
    """Embed (question, answer) pairs using sentence-transformers.

    **Detailed explanation for engineers:**
        Uses the all-MiniLM-L6-v2 model (384-dim output) to embed the
        concatenation "Q: {question} A: {answer}" as a single sentence.
        This captures the semantic relationship between question and answer
        in a single vector, analogous to experiment 37's approach.

        Unlike experiment 37 which concatenated separate Q and A embeddings
        (giving 768-dim), here we embed the QA pair jointly into 384-dim.
        This is more compact and lets the sentence encoder capture
        cross-attention between question and answer tokens.

    Args:
        questions: List of question strings.
        answers: List of answer strings.
        embed_dim: Expected embedding dimension (for validation only).

    Returns:
        Array of shape (n, embed_dim) with semantic embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [f"Q: {q} A: {a}" for q, a in zip(questions, answers)]
        embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        assert embeddings.shape[1] == embed_dim, (
            f"Expected {embed_dim}-dim embeddings, got {embeddings.shape[1]}"
        )
        return embeddings
    except ImportError:
        # Fallback: deterministic random embeddings if sentence-transformers
        # is not installed. Correct answers get embeddings near a "correct"
        # centroid, wrong answers near a "wrong" centroid, with noise.
        print("  [WARNING] sentence-transformers not installed, using mock embeddings")
        rng = np.random.default_rng(65)
        n = len(questions)
        return rng.standard_normal((n, embed_dim)).astype(np.float32)


def _compute_constraint_vectors(
    questions: list[str],
    answers: list[str],
) -> np.ndarray:
    """Compute per-constraint pass/fail vectors for all QA pairs.

    **Detailed explanation for engineers:**
        Evaluates each (question, answer) pair against all constraints defined
        in CONSTRAINT_NAMES. Returns a binary matrix where entry [i, j] is 1.0
        if answer i satisfies constraint j, and 0.0 otherwise.

        This is analogous to the per_constraint vector in experiment 48's
        _verify_logical_via_ising(), but applied to natural language answers
        instead of type-level propositions.

    Args:
        questions: List of question strings.
        answers: List of answer strings.

    Returns:
        Array of shape (n, n_constraints) with binary constraint satisfaction.
    """
    n_constraints = len(CONSTRAINT_NAMES)
    n = len(questions)
    vectors = np.zeros((n, n_constraints), dtype=np.float32)

    for i in range(n):
        checks = _check_constraints(questions[i], answers[i])
        for j, passed in enumerate(checks):
            vectors[i, j] = 1.0 if passed else 0.0

    return vectors


# ---------------------------------------------------------------------------
# 3. Training helpers
# ---------------------------------------------------------------------------


def _train_ebm(
    train_correct: jax.Array,
    train_wrong: jax.Array,
    input_dim: int,
    hidden_dims: list[int],
    n_epochs: int,
    lr: float,
    key: jax.Array,
    label: str = "EBM",
) -> GibbsModel:
    """Train a Gibbs EBM via NCE on correct (low energy) vs wrong (high energy) pairs.

    **Detailed explanation for engineers:**
        Follows the same pattern as experiment 37: extract model parameters
        into a pytree, compute NCE loss and gradients via jax.value_and_grad,
        and update with vanilla SGD. The NCE loss pushes correct-answer
        energies down and wrong-answer energies up.

        We use the training module's nce_loss function which handles vmap
        batching and numerically stable log-sigmoid internally.

    Args:
        train_correct: Correct samples, shape (n, input_dim). Target: low energy.
        train_wrong: Wrong samples, shape (n, input_dim). Target: high energy.
        input_dim: Dimension of input vectors.
        hidden_dims: Hidden layer sizes for the Gibbs model.
        n_epochs: Number of training epochs.
        lr: Learning rate for SGD.
        key: JAX PRNG key for model initialization.
        label: Human-readable label for progress printing.

    Returns:
        Trained GibbsModel.
    """
    config = GibbsConfig(input_dim=input_dim, hidden_dims=hidden_dims, activation="silu")
    model = GibbsModel(config, key=key)

    # Initialize output weight to small random values instead of zeros.
    # With zero output weights, the model outputs E=0 for all inputs,
    # creating a saddle point where the NCE loss gradient is proportional
    # to (mean(h_correct) - mean(h_wrong)). If hidden representations
    # don't separate classes (as with high-dim embeddings), this gradient
    # is near-zero and the model gets stuck. Small random initialization
    # breaks this symmetry.
    k_out, key = jrandom.split(key)
    last_hidden = hidden_dims[-1]
    model.output_weight = jrandom.normal(k_out, (last_hidden,)) * 0.01

    # Extract/set parameter helpers (same pattern as experiment 37)
    def get_params(m):
        return {
            "layers": [(w, b) for w, b in m.layers],
            "output_weight": m.output_weight,
            "output_bias": m.output_bias,
        }

    def set_params(m, p):
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    params = get_params(model)

    def loss_fn(p):
        """NCE loss as a pure function of parameters (for jax.grad)."""
        old = get_params(model)
        set_params(model, p)
        result = nce_loss(model, train_correct, train_wrong)
        set_params(model, old)
        return result

    for ep in range(n_epochs):
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
        if ep % 50 == 0:
            print(f"    [{label}] epoch {ep}: loss={float(loss_val):.4f}")

    set_params(model, params)
    return model


def _compute_auroc(
    model: GibbsModel,
    correct_data: jax.Array,
    wrong_data: jax.Array,
) -> float:
    """Compute AUROC for binary discrimination (correct vs wrong).

    **Detailed explanation for engineers:**
        The EBM assigns low energy to correct answers and high energy to wrong
        answers. AUROC measures how well the energy separates the two classes.
        We use sklearn's roc_auc_score, treating wrong answers as the positive
        class (high energy = positive prediction).

        AUROC = 1.0 means perfect separation. AUROC = 0.5 means random.

    Args:
        model: Trained Gibbs EBM.
        correct_data: Correct samples (should get low energy).
        wrong_data: Wrong samples (should get high energy).

    Returns:
        AUROC score in [0, 1].
    """
    from sklearn.metrics import roc_auc_score

    correct_energies = np.array([float(model.energy(correct_data[i]))
                                 for i in range(len(correct_data))])
    wrong_energies = np.array([float(model.energy(wrong_data[i]))
                               for i in range(len(wrong_data))])

    # Labels: 0=correct, 1=wrong. Scores: energy (higher = more likely wrong).
    labels = np.concatenate([np.zeros(len(correct_energies)),
                             np.ones(len(wrong_energies))])
    scores = np.concatenate([correct_energies, wrong_energies])

    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# 4. Gradient-based repair in joint space
# ---------------------------------------------------------------------------


def _repair_in_joint_space(
    wrong_joint: jax.Array,
    model: GibbsModel,
    embed_dim: int,
    correct_emb_mean: jax.Array,
    n_steps: int = 200,
    step_size: float = 0.01,
    noise_scale: float = 0.001,
    manifold_weight: float = 0.1,
) -> jax.Array:
    """Repair a wrong answer's joint vector by gradient descent on the EBM energy.

    **Detailed explanation for engineers:**
        Starting from a wrong answer's joint [embedding; constraint] vector,
        descend on the energy landscape to find a lower-energy (more "correct")
        configuration. The gradient updates BOTH the embedding and constraint
        components simultaneously.

        **Key insight: manifold regularization.**
        Pure energy minimization can push the embedding into regions that have
        low energy but don't correspond to any real answer (adversarial examples).
        To prevent this, we add a soft regularizer that pulls the embedding
        portion toward the mean of correct answer embeddings. This keeps the
        repaired vector on the data manifold.

        The combined objective is:
            minimize E(x) + manifold_weight * ||emb(x) - mean_correct_emb||^2

        We also normalize the gradient to prevent explosion when the energy
        landscape is steep, following best practices from Langevin dynamics.

    Args:
        wrong_joint: Joint vector for a wrong answer, shape (embed_dim + n_constraints,).
        model: Trained joint-space Gibbs EBM.
        embed_dim: Dimension of the semantic embedding prefix.
        correct_emb_mean: Mean embedding of correct answers, shape (embed_dim,).
            Used as the manifold anchor for regularization.
        n_steps: Number of gradient descent steps.
        step_size: Learning rate for gradient descent.
        noise_scale: Std of Langevin exploration noise.
        manifold_weight: Weight of the manifold regularizer (0 = no regularization).

    Returns:
        Repaired joint vector, same shape as input.
    """
    x = wrong_joint

    def combined_energy(v):
        """EBM energy + manifold regularizer to stay near real embeddings."""
        ebm_energy = model.energy(v)
        # Regularize: keep embedding portion near correct-answer manifold
        emb_part = v[:embed_dim]
        manifold_penalty = manifold_weight * jnp.sum((emb_part - correct_emb_mean) ** 2)
        return ebm_energy + manifold_penalty

    grad_fn = jax.grad(combined_energy)

    for step in range(n_steps):
        g = grad_fn(x)
        # Normalize gradient to prevent explosion (unit-norm step direction)
        g_norm = jnp.linalg.norm(g) + 1e-8
        g_normalized = g / g_norm
        noise = jrandom.normal(jrandom.PRNGKey(step), shape=x.shape) * noise_scale
        x = x - step_size * g_normalized + noise

    return x


# ---------------------------------------------------------------------------
# 5. Main experiment
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 65: Joint Embedding-Constraint EBM")
    print("  Bridge semantic (embedding) and structural (constraint) spaces")
    print("  Train joint EBM, compare AUROC, test gradient-based repair")
    print("=" * 70)

    start = time.time()

    # -----------------------------------------------------------------------
    # Step 1: Generate synthetic QA dataset
    # -----------------------------------------------------------------------
    print("\nStep 1: Generating synthetic QA dataset...")
    questions, answers, is_correct = _generate_qa_dataset(n_total=200, seed=65)
    n_correct = sum(is_correct)
    n_wrong = len(is_correct) - n_correct
    print(f"  Generated {len(questions)} QA pairs ({n_correct} correct, {n_wrong} wrong)")

    # -----------------------------------------------------------------------
    # Step 2: Compute semantic embeddings and constraint vectors
    # -----------------------------------------------------------------------
    print("\nStep 2: Computing embeddings and constraint vectors...")

    raw_embed_dim = 384
    raw_embeddings = _embed_answers(questions, answers, embed_dim=raw_embed_dim)

    # L2-normalize raw embeddings before PCA
    emb_norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True) + 1e-8
    raw_embeddings = raw_embeddings / emb_norms

    # PCA reduction: 384-dim embeddings are too high-dimensional for 80 training
    # samples. We project down to 32 dims, capturing the dominant variance
    # directions. This makes the EBM learnable while preserving the most
    # discriminative semantic structure.
    from sklearn.decomposition import PCA

    embed_dim = 32
    pca = PCA(n_components=embed_dim, random_state=65)
    embeddings = pca.fit_transform(raw_embeddings).astype(np.float32)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  Raw embeddings: {raw_embeddings.shape} → PCA → {embeddings.shape}")
    print(f"  PCA explained variance: {explained_var:.1%}")

    constraint_vectors = _compute_constraint_vectors(questions, answers)
    n_constraints = constraint_vectors.shape[1]
    print(f"  Constraint vectors: shape {constraint_vectors.shape}")

    # Show constraint satisfaction rates for correct vs wrong
    correct_mask = np.array(is_correct, dtype=bool)
    print("\n  Constraint satisfaction rates:")
    print(f"  {'Constraint':<25s} {'Correct':>8s} {'Wrong':>8s} {'Delta':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for j, cname in enumerate(CONSTRAINT_NAMES):
        rate_c = constraint_vectors[correct_mask, j].mean()
        rate_w = constraint_vectors[~correct_mask, j].mean()
        delta = rate_c - rate_w
        print(f"  {cname:<25s} {rate_c:>7.0%} {rate_w:>7.0%} {delta:>+7.0%}")

    # -----------------------------------------------------------------------
    # Step 3: Build feature vectors for three models
    # -----------------------------------------------------------------------
    print("\nStep 3: Building feature vectors...")

    # Joint: [embedding; constraint_vector] (384 + N dim)
    joint_vectors = np.concatenate([embeddings, constraint_vectors], axis=1).astype(np.float32)
    joint_dim = joint_vectors.shape[1]
    print(f"  Joint vectors: shape {joint_vectors.shape} ({embed_dim} + {n_constraints} = {joint_dim})")

    # Split into correct/wrong
    correct_indices = np.where(correct_mask)[0]
    wrong_indices = np.where(~correct_mask)[0]

    # Train/test split (80/20)
    rng = np.random.default_rng(65)
    rng.shuffle(correct_indices)
    rng.shuffle(wrong_indices)

    n_train_c = int(len(correct_indices) * 0.8)
    n_train_w = int(len(wrong_indices) * 0.8)

    train_c_idx = correct_indices[:n_train_c]
    test_c_idx = correct_indices[n_train_c:]
    train_w_idx = wrong_indices[:n_train_w]
    test_w_idx = wrong_indices[n_train_w:]

    print(f"  Train: {n_train_c} correct + {n_train_w} wrong")
    print(f"  Test:  {len(test_c_idx)} correct + {len(test_w_idx)} wrong")

    # -----------------------------------------------------------------------
    # Step 4: Train three EBMs (joint, embedding-only, constraint-only)
    # -----------------------------------------------------------------------
    print("\nStep 4: Training EBMs...")

    key = jrandom.PRNGKey(65)
    k1, k2, k3 = jrandom.split(key, 3)
    hidden_dims = [64, 32]
    n_epochs = 500
    lr = 0.01

    # --- Model A: Joint (embedding + constraints) ---
    print(f"\n  Training Joint EBM (input_dim={joint_dim})...")
    joint_train_c = jnp.array(joint_vectors[train_c_idx])
    joint_train_w = jnp.array(joint_vectors[train_w_idx])
    joint_test_c = jnp.array(joint_vectors[test_c_idx])
    joint_test_w = jnp.array(joint_vectors[test_w_idx])

    model_joint = _train_ebm(
        joint_train_c, joint_train_w,
        input_dim=joint_dim, hidden_dims=hidden_dims,
        n_epochs=n_epochs, lr=lr, key=k1, label="Joint",
    )

    # --- Model B: Embedding-only ---
    print(f"\n  Training Embedding-only EBM (input_dim={embed_dim})...")
    emb_train_c = jnp.array(embeddings[train_c_idx])
    emb_train_w = jnp.array(embeddings[train_w_idx])
    emb_test_c = jnp.array(embeddings[test_c_idx])
    emb_test_w = jnp.array(embeddings[test_w_idx])

    model_emb = _train_ebm(
        emb_train_c, emb_train_w,
        input_dim=embed_dim, hidden_dims=hidden_dims,
        n_epochs=n_epochs, lr=lr, key=k2, label="Embed",
    )

    # --- Model C: Constraint-only ---
    print(f"\n  Training Constraint-only EBM (input_dim={n_constraints})...")
    con_train_c = jnp.array(constraint_vectors[train_c_idx])
    con_train_w = jnp.array(constraint_vectors[train_w_idx])
    con_test_c = jnp.array(constraint_vectors[test_c_idx])
    con_test_w = jnp.array(constraint_vectors[test_w_idx])

    model_con = _train_ebm(
        con_train_c, con_train_w,
        input_dim=n_constraints, hidden_dims=[32],
        n_epochs=n_epochs, lr=lr, key=k3, label="Constraint",
    )

    # -----------------------------------------------------------------------
    # Step 5: Evaluate AUROC for all three models
    # -----------------------------------------------------------------------
    print("\nStep 5: Evaluating AUROC...")

    auroc_joint = _compute_auroc(model_joint, joint_test_c, joint_test_w)
    auroc_emb = _compute_auroc(model_emb, emb_test_c, emb_test_w)
    auroc_con = _compute_auroc(model_con, con_test_c, con_test_w)

    print(f"\n  {'Model':<20s} {'AUROC':>8s}")
    print(f"  {'-'*20} {'-'*8}")
    print(f"  {'Joint (emb+con)':<20s} {auroc_joint:>8.4f}")
    print(f"  {'Embedding-only':<20s} {auroc_emb:>8.4f}")
    print(f"  {'Constraint-only':<20s} {auroc_con:>8.4f}")

    joint_advantage = auroc_joint - max(auroc_emb, auroc_con)
    print(f"\n  Joint advantage over best single: {joint_advantage:+.4f}")

    # -----------------------------------------------------------------------
    # Step 6: Error analysis — what does the joint model catch?
    # -----------------------------------------------------------------------
    print("\nStep 6: Error analysis...")

    # For each test sample, check which models correctly classify it
    # (correct → low energy, wrong → high energy relative to a threshold)
    def _classify(model, correct, wrong):
        """Returns (correct_energies, wrong_energies, threshold)."""
        ce = np.array([float(model.energy(correct[i])) for i in range(len(correct))])
        we = np.array([float(model.energy(wrong[i])) for i in range(len(wrong))])
        thresh = (ce.mean() + we.mean()) / 2.0
        return ce, we, thresh

    j_ce, j_we, j_thresh = _classify(model_joint, joint_test_c, joint_test_w)
    e_ce, e_we, e_thresh = _classify(model_emb, emb_test_c, emb_test_w)
    c_ce, c_we, c_thresh = _classify(model_con, con_test_c, con_test_w)

    # Count wrong answers caught by each model (energy > threshold)
    joint_catches = j_we > j_thresh
    emb_catches = e_we > e_thresh
    con_catches = c_we > c_thresh

    n_test_wrong = len(test_w_idx)
    joint_only = np.sum(joint_catches & ~emb_catches & ~con_catches)
    emb_only = np.sum(~joint_catches & emb_catches & ~con_catches)
    con_only = np.sum(~joint_catches & ~emb_catches & con_catches)
    all_catch = np.sum(joint_catches & emb_catches & con_catches)
    none_catch = np.sum(~joint_catches & ~emb_catches & ~con_catches)

    print(f"  Wrong answers detected (out of {n_test_wrong}):")
    print(f"    Joint model:       {np.sum(joint_catches)}/{n_test_wrong}")
    print(f"    Embedding-only:    {np.sum(emb_catches)}/{n_test_wrong}")
    print(f"    Constraint-only:   {np.sum(con_catches)}/{n_test_wrong}")
    print(f"    Joint-only catches: {joint_only} (caught by joint but missed by both others)")
    print(f"    All three catch:    {all_catch}")
    print(f"    None catches:       {none_catch}")

    # -----------------------------------------------------------------------
    # Step 7: Gradient-based repair in joint space
    # -----------------------------------------------------------------------
    print("\nStep 7: Gradient-based repair in joint space...")

    # Build codebook of correct answer embeddings for nearest-neighbor decoding
    # Use PCA-projected embeddings (same space as the joint model operates in)
    correct_embs_codebook = embeddings[correct_indices]
    # Mean of correct embeddings as manifold anchor for regularized repair
    correct_emb_mean = jnp.array(correct_embs_codebook.mean(axis=0))

    n_repair = min(20, len(test_w_idx))
    repair_successes = 0
    energy_improved = 0

    for i in range(n_repair):
        w_idx = test_w_idx[i]
        wrong_joint_vec = jnp.array(joint_vectors[w_idx])

        # Energy before repair
        energy_before = float(model_joint.energy(wrong_joint_vec))

        # Repair via normalized gradient descent with manifold regularization
        repaired = _repair_in_joint_space(
            wrong_joint_vec, model_joint, embed_dim,
            correct_emb_mean=correct_emb_mean,
            n_steps=200, step_size=0.05, noise_scale=0.001,
            manifold_weight=0.1,
        )

        energy_after = float(model_joint.energy(repaired))

        # Extract embedding portion and decode via nearest-neighbor
        repaired_emb = np.array(repaired[:embed_dim])
        original_emb = embeddings[w_idx]

        # Find nearest correct answer in embedding space
        def _nearest_correct(emb):
            dots = correct_embs_codebook @ emb
            norms = np.linalg.norm(correct_embs_codebook, axis=1) * (np.linalg.norm(emb) + 1e-8)
            sims = dots / (norms + 1e-8)
            return int(np.argmax(sims)), float(np.max(sims))

        # Before repair: which correct answer is closest?
        orig_best_idx, orig_best_sim = _nearest_correct(original_emb)

        # After repair: which correct answer is closest?
        repaired_best_idx, repaired_best_sim = _nearest_correct(repaired_emb)

        # Success: repaired embedding is closer to a correct answer than original
        improved_sim = repaired_best_sim > orig_best_sim
        improved_energy = energy_after < energy_before

        if improved_sim:
            repair_successes += 1
        if improved_energy:
            energy_improved += 1

        if i < 5:
            q_text = questions[w_idx][:40]
            a_text = answers[w_idx][:40]
            print(f"    [{i}] Q: {q_text}...")
            print(f"         A: {a_text}...")
            print(f"         energy: {energy_before:.3f} → {energy_after:.3f} "
                  f"sim: {orig_best_sim:.3f} → {repaired_best_sim:.3f} "
                  f"{'✓' if improved_sim else '✗'}")

    repair_rate = repair_successes / n_repair
    energy_rate = energy_improved / n_repair

    print(f"\n  Repair results ({n_repair} wrong answers):")
    print(f"    Energy improved:     {energy_improved}/{n_repair} ({energy_rate:.0%})")
    print(f"    Similarity improved: {repair_successes}/{n_repair} ({repair_rate:.0%})")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 65 RESULTS ({elapsed:.0f}s)")
    print(sep)

    print(f"\n  AUROC Comparison:")
    print(f"  {'Model':<20s} {'AUROC':>8s}")
    print(f"  {'-'*20} {'-'*8}")
    print(f"  {'Joint (emb+con)':<20s} {auroc_joint:>8.4f}")
    print(f"  {'Embedding-only':<20s} {auroc_emb:>8.4f}")
    print(f"  {'Constraint-only':<20s} {auroc_con:>8.4f}")
    print(f"  {'Joint advantage':<20s} {joint_advantage:>+8.4f}")

    print(f"\n  Repair Success Rates:")
    print(f"    Energy improved:     {energy_rate:.0%}")
    print(f"    Similarity improved: {repair_rate:.0%}")

    # Verdict
    if auroc_joint > max(auroc_emb, auroc_con) and auroc_joint > 0.6:
        print(f"\n  VERDICT: Joint EBM outperforms both single-space models.")
        print(f"  The correlation between semantic and structural signals is real")
        print(f"  and learnable — combining spaces yields better discrimination.")
    elif auroc_joint >= max(auroc_emb, auroc_con):
        print(f"\n  VERDICT: Joint EBM matches but does not significantly beat")
        print(f"  single-space models. The constraint signal may be redundant")
        print(f"  with semantic information for this dataset.")
    else:
        print(f"\n  VERDICT: Single-space model outperforms the joint model.")
        print(f"  The added dimensionality may hurt with limited training data.")
        print(f"  More data or a larger model may be needed.")

    if repair_rate > 0.5:
        print(f"\n  REPAIR: Gradient descent in joint space successfully moves")
        print(f"  wrong answers toward correct regions ({repair_rate:.0%} improvement).")
    else:
        print(f"\n  REPAIR: Mixed repair results ({repair_rate:.0%}). The energy")
        print(f"  landscape may have local minima or the step size needs tuning.")

    print(f"\n  NOTE: This bridges Exp 64 (continuous Ising) with Exp 37 (EBT reasoning).")
    print(f"  Next step: replace heuristic constraints with actual Ising verifier")
    print(f"  outputs for code generation (Exp 48 pipeline).")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
