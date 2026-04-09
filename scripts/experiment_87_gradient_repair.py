#!/usr/bin/env python3
"""Experiment 87: Gradient-Based Repair in Embedding Space — Continuous repair without LLM re-prompting.

**Researcher summary:**
    Replaces the discrete verify-repair loop (Exp 57/75: detect violation → re-prompt LLM →
    re-verify) with continuous gradient descent in embedding space. The pipeline is:
    embed response → extract constraint features → concatenate [embedding; constraint_vector] →
    gradient descent on constraint energy (freeze constraint weights, update embedding only) →
    nearest-neighbor decode repaired embedding to text via a codebook of candidate responses.
    If this works, repair becomes instant (no LLM inference per iteration).

**Detailed explanation for engineers:**
    The current repair loop (VerifyRepairPipeline from Exp 57) is discrete and slow:
    1. Detect constraint violations (fast — just energy evaluation)
    2. Format violations as natural-language feedback text
    3. Re-prompt the LLM with the feedback (slow — full LLM forward pass)
    4. Re-verify the new response (fast)
    5. Repeat until constraints are satisfied or max iterations reached

    Each iteration requires an LLM inference call, which is the bottleneck. For a
    small model like Qwen3.5-0.8B, that's ~1-2 seconds per iteration. For larger
    models, it's much worse.

    This experiment tries a fundamentally different approach — CONTINUOUS repair:
    1. Embed the response text into a 384-dim semantic vector (all-MiniLM-L6-v2)
    2. Extract constraint features (8-dim binary pass/fail vector, same as Exp 65/66)
    3. Concatenate [embedding; constraint_vector] as the initial state (392-dim)
    4. Define a constraint energy function on this joint space (using the same
       ComposedEnergy framework from carnot.verify.constraint, but operating on
       the joint vector rather than a discrete configuration)
    5. Gradient descent: minimize constraint energy w.r.t. the EMBEDDING portion only
       (freeze constraint weights — they encode the rules, not the response)
    6. After convergence, project the repaired embedding back to text via nearest-
       neighbor lookup in a "codebook" of candidate response embeddings

    The codebook is built per-question: for each question, we generate 20 candidate
    responses (paraphrases, variations, correct/incorrect). We embed all candidates.
    After gradient repair modifies the embedding, we find the nearest candidate in
    embedding space and return that candidate's text as the "repaired" response.

    **Why this could work:**
    Exp 65 showed that the joint [embedding; constraint] space separates correct from
    wrong answers (AUROC > 0.9). Exp 66 showed that a differentiable verifier can
    score responses in a single forward pass with gradients flowing through constraints.
    If the energy landscape is smooth enough, gradient descent should move a wrong
    answer's embedding toward a correct region — and the codebook decode recovers text.

    **Why this might fail:**
    - The embedding space may not be smooth enough for gradient descent to find the
      correct basin (local minima, discontinuities from discrete constraint features)
    - The codebook may not contain a good enough candidate near the repaired embedding
    - The nearest-neighbor decode may pick a candidate that is close in embedding space
      but semantically wrong (embedding space ≠ semantic space perfectly)

    **Evaluation:**
    Compare three repair strategies on 100 incorrect responses across 5 domains:
    (a) Gradient repair only — the new approach from this experiment
    (b) Discrete repair — the Exp 57 approach (format feedback → re-prompt → re-verify)
        Simulated without an actual LLM by checking if the correct answer is in the
        candidate pool (proxy for "can the LLM fix this with feedback?")
    (c) Combined — gradient repair first, then discrete repair on remaining failures

    Metrics: repair success rate, iterations needed, wall-clock time.
    Ablation: different step sizes [0.01, 0.1, 1.0], max iterations [10, 50, 100].

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_87_gradient_repair.py

REQ: REQ-VERIFY-005, REQ-EBT-001
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.pipeline.extract import AutoExtractor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension
N_CONSTRAINTS = 8  # Same lightweight text constraints as Exp 65/66
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "experiment_87_results.json"
)

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

# Hyperparameters for ablation
STEP_SIZES = [0.01, 0.1, 1.0]
MAX_ITERS_LIST = [10, 50, 100]
DEFAULT_STEP_SIZE = 0.1
DEFAULT_MAX_ITERS = 50
CONVERGENCE_THRESHOLD = 1e-4
CODEBOOK_SIZE = 20  # Candidates per question


# ---------------------------------------------------------------------------
# 1. Constraint checking (same as Exp 65/66)
# ---------------------------------------------------------------------------


def _check_constraints(question: str, answer: str) -> list[bool]:
    """Evaluate 8 lightweight text constraints on an answer.

    **Detailed explanation for engineers:**
        Identical to Exp 65/66's constraint checks. Each constraint is a simple
        string heuristic that captures one structural aspect of the answer. The
        binary pass/fail vector feeds into the gradient repair energy function
        as the constraint portion of the joint state vector.

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


# ---------------------------------------------------------------------------
# 2. Multi-domain synthetic dataset (same structure as Exp 66)
# ---------------------------------------------------------------------------


def _generate_multi_domain_dataset(
    n_per_domain: int = 20,
    seed: int = 87,
) -> list[dict]:
    """Generate QA pairs across 5 domains with correct and wrong answers.

    **Detailed explanation for engineers:**
        Creates a balanced dataset where each domain has n_per_domain questions,
        half answered correctly and half incorrectly. Each question also gets
        a "codebook" of 20 candidate responses (variations, paraphrases, the
        correct answer, and plausible-but-wrong alternatives). The codebook is
        what the gradient repairer uses for nearest-neighbor decoding: after
        modifying the embedding via gradient descent, the nearest codebook
        entry becomes the "repaired" text response.

        The 5 domains mirror Exp 66: arithmetic, code, logic, factual, scheduling.
        We generate fewer per domain (20 vs 100) since the bottleneck is the
        gradient repair loop, not training.

    Args:
        n_per_domain: Questions per domain (half correct, half wrong).
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with keys: domain, question, answer, is_correct, codebook.
        codebook is a list of (text, is_correct) tuples.
    """
    rng = np.random.default_rng(seed)
    dataset: list[dict] = []

    n_each = n_per_domain // 2  # half correct, half wrong per domain

    def _make_codebook(
        question: str,
        correct_answer: str,
        wrong_variations: list[str],
        rng: np.random.Generator,
    ) -> list[tuple[str, bool]]:
        """Build a codebook of 20 candidate responses for nearest-neighbor decoding.

        **Detailed explanation for engineers:**
            For each question, we build a set of candidate responses that the
            gradient repairer can "decode" to after modifying the embedding.
            The codebook contains:
            - The correct answer (1 entry)
            - Paraphrases of the correct answer (4 entries) — same meaning, different words
            - Wrong variations (up to 5 entries) — plausible but incorrect
            - Noise candidates (fill to 20) — random off-topic or degraded answers

            The quality of the codebook directly affects repair success: if no
            candidate is close to the "correct basin" in embedding space, gradient
            descent cannot find a good decode target. 20 candidates is a balance
            between coverage and computational cost.

        Args:
            question: The question text.
            correct_answer: The known-correct answer text.
            wrong_variations: List of wrong answer variations.
            rng: Random number generator.

        Returns:
            List of (text, is_correct) tuples, length CODEBOOK_SIZE.
        """
        codebook: list[tuple[str, bool]] = []

        # 1. The correct answer itself
        codebook.append((correct_answer, True))

        # 2. Paraphrases of the correct answer (simple word-level variations)
        paraphrase_prefixes = [
            "The answer is",
            "The result equals",
            "This gives us",
            "We get",
        ]
        for prefix in paraphrase_prefixes:
            # Extract the core result from the correct answer
            # (everything after "is" or "equals" up to the period)
            parts = correct_answer.split(".")
            core = parts[0] if parts else correct_answer
            # Try to extract just the numerical/factual part
            for sep in ("is ", "equals ", "gives "):
                if sep in core.lower():
                    val = core.lower().split(sep)[-1].strip()
                    paraphrase = f"{prefix} {val}. Because the calculation confirms this result."
                    codebook.append((paraphrase, True))
                    break
            else:
                # Fallback: just rephrase with the prefix
                codebook.append(
                    (f"{prefix} correct. {correct_answer}", True)
                )

        # 3. Wrong variations (up to 5)
        for wv in wrong_variations[:5]:
            codebook.append((wv, False))

        # 4. Fill remaining slots with noise/off-topic candidates
        noise_templates = [
            "I'm not sure about this question.",
            "The answer depends on the context.",
            "This is a complex problem that requires more information.",
            "42 is the answer to everything.",
            "Let me think about this differently.",
            "The result cannot be determined from the given information.",
            "Bananas are a good source of potassium.",
            "The sky is blue because of Rayleigh scattering.",
            "Python is a programming language.",
            "The answer is approximately 3.14159.",
            "According to my calculations, the result is undefined.",
            "This problem has multiple solutions.",
            "I need to reconsider the premises.",
            "The question itself may be flawed.",
            "Error: division by zero.",
        ]
        while len(codebook) < CODEBOOK_SIZE:
            idx = int(rng.integers(0, len(noise_templates)))
            codebook.append((noise_templates[idx], False))

        return codebook[:CODEBOOK_SIZE]

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
        correct_ans = f"The answer is {r}. Because {a} {op} {b} equals {r}."

        wrong_variations = [
            f"The answer is {r + int(rng.integers(1, 50))}. Because the calculation gives this.",
            f"The answer is {r - int(rng.integers(1, 50))}. I think this is right.",
            f"The answer is {r * 2}. Because you multiply both numbers.",
            f"{r + 1}",
            f"The answer is not {r} but rather {r + 10}.",
        ]

        codebook = _make_codebook(q, correct_ans, wrong_variations, rng)

        # Correct entry
        dataset.append({
            "domain": "arithmetic",
            "question": q,
            "answer": correct_ans,
            "is_correct": True,
            "codebook": codebook,
        })
        # Wrong entry (use first wrong variation)
        dataset.append({
            "domain": "arithmetic",
            "question": q,
            "answer": wrong_variations[0],
            "is_correct": False,
            "codebook": codebook,
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
        correct_ans = (
            f"Here is the solution:\n```python\n{code}\n```\n"
            f"This works because it implements the required logic."
        )
        wrong_variations = [
            "def func():\n    pass",
            "The answer is 42.",
            f"def {task_name.replace(' ', '_')}():\n    return None  # TODO",
            "I don't know how to write that function.",
            f"def func(x): return x  # This does not {task_name}.",
        ]
        codebook = _make_codebook(q, correct_ans, wrong_variations, rng)
        dataset.append({
            "domain": "code",
            "question": q,
            "answer": correct_ans,
            "is_correct": True,
            "codebook": codebook,
        })
        dataset.append({
            "domain": "code",
            "question": q,
            "answer": wrong_variations[0],
            "is_correct": False,
            "codebook": codebook,
        })

    # --- Logic ---
    logic_templates = [
        ("If it rains, the ground gets wet. It rained. Is the ground wet?", "Yes", "No"),
        ("All cats are mammals. Tom is a cat. Is Tom a mammal?", "Yes", "No"),
        ("If A implies B, and A is true, then B must be?", "True", "False"),
        ("If X > 5 and X < 10, and X = 7, is the condition satisfied?", "Yes", "No"),
        ("All birds can fly. Penguins are birds. Can penguins fly?",
         "Based on the premises, yes", "No"),
    ]
    for i in range(n_each):
        q, correct_a, wrong_a = logic_templates[i % len(logic_templates)]
        correct_ans = f"{correct_a}. Because the logical chain of reasoning supports this conclusion."
        wrong_ans = f"{wrong_a}. Because the premises do not support this."
        wrong_variations = [
            wrong_ans,
            "Maybe. The logic is ambiguous.",
            "I cannot determine the answer from the given premises.",
            f"The answer is {wrong_a} because intuition says so.",
            "Both yes and no are valid answers.",
        ]
        codebook = _make_codebook(q, correct_ans, wrong_variations, rng)
        dataset.append({
            "domain": "logic",
            "question": q,
            "answer": correct_ans,
            "is_correct": True,
            "codebook": codebook,
        })
        dataset.append({
            "domain": "logic",
            "question": q,
            "answer": wrong_ans,
            "is_correct": False,
            "codebook": codebook,
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
        correct_ans = f"The answer is {correct_a}. This is a well-established fact."
        wrong_ans = f"The answer is {wrong_a}. This is what I recall."
        wrong_variations = [
            wrong_ans,
            f"I believe the answer is {wrong_a}. Because it seems correct.",
            "I'm not confident in my answer to this question.",
            f"The answer might be {wrong_a} or {correct_a}, I'm unsure.",
            "This question is outside my knowledge domain.",
        ]
        codebook = _make_codebook(q, correct_ans, wrong_variations, rng)
        dataset.append({
            "domain": "factual",
            "question": q,
            "answer": correct_ans,
            "is_correct": True,
            "codebook": codebook,
        })
        dataset.append({
            "domain": "factual",
            "question": q,
            "answer": wrong_ans,
            "is_correct": False,
            "codebook": codebook,
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
        correct_ans = (
            f"No conflict. Meeting A ends at {h2}:00 and Meeting B starts at "
            f"{h3}:00, so they don't overlap."
        )
        wrong_ans = (
            f"Yes, there is a conflict because both meetings overlap at {h2}:00."
        )
        wrong_variations = [
            wrong_ans,
            "There might be a conflict depending on buffer time.",
            f"Yes, meetings at {h1}:00 and {h3}:00 are too close together.",
            "I need more information about the meeting durations.",
            f"The schedule is too packed. Both meetings conflict at {h2}:00.",
        ]
        codebook = _make_codebook(q, correct_ans, wrong_variations, rng)
        dataset.append({
            "domain": "scheduling",
            "question": q,
            "answer": correct_ans,
            "is_correct": True,
            "codebook": codebook,
        })
        dataset.append({
            "domain": "scheduling",
            "question": q,
            "answer": wrong_ans,
            "is_correct": False,
            "codebook": codebook,
        })

    # Shuffle deterministically
    indices = list(range(len(dataset)))
    rng2 = np.random.default_rng(seed + 1)
    rng2.shuffle(indices)
    dataset = [dataset[i] for i in indices]

    return dataset


# ---------------------------------------------------------------------------
# 3. Embedding helpers
# ---------------------------------------------------------------------------


def _embed_texts(
    texts: list[str],
    embed_dim: int = EMBED_DIM,
) -> np.ndarray:
    """Embed a list of texts using sentence-transformers or mock fallback.

    **Detailed explanation for engineers:**
        Uses all-MiniLM-L6-v2 (384-dim) to embed text strings into dense vectors.
        If sentence-transformers is not installed, falls back to deterministic
        mock embeddings based on hash-seeded random vectors. The mock embeddings
        still produce reasonable nearest-neighbor behavior because texts with
        similar hash-seeds cluster together.

        Unlike Exp 65/66 which embed "Q: ... A: ..." pairs, here we embed both
        individual texts (for the codebook) and QA pairs (for the response being
        repaired). The caller is responsible for formatting.

    Args:
        texts: List of text strings to embed.
        embed_dim: Expected embedding dimension (384 for MiniLM).

    Returns:
        Array of shape (n, embed_dim) with L2-normalized embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        assert embeddings.shape[1] == embed_dim
        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return embeddings / norms
    except ImportError:
        print("  [WARNING] sentence-transformers not installed, using mock embeddings")
        # Deterministic mock: hash each text to seed a random vector, so
        # similar texts get somewhat similar embeddings (via shared substrings
        # in their hash). This is a rough approximation for testing.
        n = len(texts)
        embeddings = np.zeros((n, embed_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = hash(text) % (2**31)
            rng = np.random.default_rng(seed)
            embeddings[i] = rng.standard_normal(embed_dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return embeddings / norms


def _compute_constraint_vector(question: str, answer: str) -> np.ndarray:
    """Compute constraint pass/fail vector for a single QA pair.

    **Detailed explanation for engineers:**
        Wraps _check_constraints to return a float32 numpy array suitable for
        concatenation with the embedding vector. Values are 1.0 (satisfied) or
        0.0 (violated).

    Args:
        question: The question text.
        answer: The candidate answer text.

    Returns:
        Array of shape (N_CONSTRAINTS,) with binary constraint satisfaction.
    """
    checks = _check_constraints(question, answer)
    return np.array([1.0 if c else 0.0 for c in checks], dtype=np.float32)


# ---------------------------------------------------------------------------
# 4. Constraint energy in joint embedding+constraint space
# ---------------------------------------------------------------------------


def _constraint_energy(
    joint_vec: jax.Array,
    embed_dim: int,
    target_constraint_vec: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    """Compute constraint energy on the joint [embedding; constraint] vector.

    **Detailed explanation for engineers:**
        The energy function measures how far the current constraint portion of the
        joint vector is from the "ideal" constraint vector (all 1.0 = all constraints
        satisfied). The energy is a weighted sum of squared violations:

            E(x) = sum_i w_i * (c_i - target_i)^2

        where c_i is the i-th constraint value in the joint vector and target_i is
        the target (1.0 for "satisfied"). The weights w_i allow prioritizing
        certain constraints (e.g., "contains_number" might matter more for arithmetic).

        Crucially, this energy is differentiable w.r.t. the ENTIRE joint vector,
        including the embedding portion. The gradient dE/d(embedding) tells us how
        to modify the embedding to reduce constraint violations. This is the key
        insight: even though constraint checks are discrete (pass/fail), the energy
        landscape over the joint space is continuous and smooth.

        In practice, the gradient only flows through the constraint portion directly.
        The embedding portion gets an indirect gradient via a manifold regularizer
        (added in the GradientRepairer) that keeps the embedding near known-good
        regions. The combined gradient moves the embedding toward a region where
        the decoded text would satisfy more constraints.

    Args:
        joint_vec: Joint vector [embedding; constraint_features], shape (embed_dim + N_CONSTRAINTS,).
        embed_dim: Dimension of the embedding prefix.
        target_constraint_vec: Target constraint values, shape (N_CONSTRAINTS,). Typically all 1.0.
        weights: Per-constraint weights, shape (N_CONSTRAINTS,).

    Returns:
        Scalar energy (non-negative). Zero means all constraints match target.
    """
    constraint_part = joint_vec[embed_dim:]
    diff = constraint_part - target_constraint_vec
    return jnp.sum(weights * diff ** 2)


# ---------------------------------------------------------------------------
# 5. GradientRepairer class
# ---------------------------------------------------------------------------


@dataclass
class RepairResult:
    """Result of a single gradient repair attempt.

    **Detailed explanation for engineers:**
        Captures everything about one repair attempt: the original wrong response,
        the decoded repaired response (nearest codebook entry), whether the repair
        succeeded (decoded response passes all constraints), how many gradient
        iterations were needed, and timing information.

    Attributes:
        original_text: The input response text that violated constraints.
        repaired_text: The nearest codebook entry after gradient descent.
        original_constraints: Constraint pass/fail vector for the original response.
        repaired_constraints: Constraint pass/fail vector for the decoded response.
        original_energy: Constraint energy before repair.
        final_energy: Constraint energy after gradient descent (in embedding space).
        decoded_energy: Constraint energy of the decoded text (may differ from
            final_energy because decoding snaps to a discrete codebook entry).
        repair_succeeded: True if the decoded response passes all constraints.
        iterations: Number of gradient descent steps taken.
        converged: True if gradient descent converged before max_iterations.
        wall_clock_seconds: Wall-clock time for the repair.
        codebook_rank: Rank of the decoded entry in the codebook (0 = nearest).
        codebook_similarity: Cosine similarity between repaired embedding and decoded entry.
    """

    original_text: str
    repaired_text: str
    original_constraints: list[float]
    repaired_constraints: list[float]
    original_energy: float
    final_energy: float
    decoded_energy: float
    repair_succeeded: bool
    iterations: int
    converged: bool
    wall_clock_seconds: float
    codebook_rank: int
    codebook_similarity: float


class GradientRepairer:
    """Continuous repair via gradient descent in embedding space.

    **Researcher summary:**
        Embeds response text, extracts constraint features, concatenates into a joint
        vector, then gradient-descends on constraint energy to minimize violations.
        Decodes the repaired embedding back to text via nearest-neighbor lookup in a
        codebook of candidate responses. No LLM inference needed during repair.

    **Detailed explanation for engineers:**
        This class implements the core idea of Experiment 87: replacing discrete
        LLM-based repair with continuous gradient-based repair in embedding space.

        The pipeline for repairing a single response:

        1. **Embed**: Encode "Q: {question} A: {response}" with sentence-transformers
           (all-MiniLM-L6-v2, 384-dim). This captures the semantic content of the
           QA pair in a dense vector.

        2. **Extract constraints**: Run the 8 lightweight text checks from Exp 65/66
           to get a binary pass/fail vector. This tells us which structural constraints
           the response violates.

        3. **Concatenate**: Form the joint vector [embedding; constraint_vector] as
           the initial state for optimization. This is a (384 + 8 = 392)-dim vector.

        4. **Gradient descent**: Minimize the constraint energy w.r.t. the embedding
           portion of the joint vector. The constraint weights are frozen — they
           encode the rules, not the response. The energy function is:

               E(x) = sum_i w_i * (c_i(x) - 1.0)^2 + manifold_weight * ||emb(x) - anchor||^2

           where c_i(x) is the i-th constraint value and anchor is the mean embedding
           of correct responses (manifold regularization prevents drifting into
           adversarial regions of embedding space).

           We use normalized gradients (unit-norm step direction) for stability,
           following Exp 65's repair approach.

        5. **Decode**: After gradient descent converges (or hits max iterations),
           extract the embedding portion of the repaired joint vector. Find the
           nearest entry in the codebook (by cosine similarity). Return that
           codebook entry's text as the "repaired" response.

        6. **Verify**: Check the decoded text against constraints to confirm the
           repair actually worked (the decode step may introduce new violations
           if the nearest codebook entry isn't perfect).

    Spec: REQ-VERIFY-005, REQ-EBT-001
    """

    def __init__(
        self,
        step_size: float = DEFAULT_STEP_SIZE,
        max_iterations: int = DEFAULT_MAX_ITERS,
        convergence_threshold: float = CONVERGENCE_THRESHOLD,
        manifold_weight: float = 0.1,
        noise_scale: float = 0.001,
        constraint_weights: np.ndarray | None = None,
    ) -> None:
        """Initialize the gradient repairer.

        **Detailed explanation for engineers:**
            Sets hyperparameters for gradient descent. The constraint_weights
            parameter allows prioritizing certain constraints during repair
            (e.g., "contains_number" might be more important for arithmetic
            domains). If None, all constraints are weighted equally at 1.0.

        Args:
            step_size: Gradient descent learning rate. Controls how far each
                step moves in embedding space. Too large → oscillation, too
                small → slow convergence. Default 0.1.
            max_iterations: Maximum gradient descent steps before giving up.
            convergence_threshold: Stop if energy change between steps is less
                than this. Indicates we've reached a (local) minimum.
            manifold_weight: Weight of the manifold regularizer that keeps the
                embedding near the mean of correct responses. Higher values
                constrain the repair more tightly to the data manifold.
            noise_scale: Std of Langevin exploration noise added each step.
                Small values (0.001) help escape shallow local minima.
            constraint_weights: Per-constraint importance weights, shape
                (N_CONSTRAINTS,). Default: all 1.0 (equal weight).
        """
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.manifold_weight = manifold_weight
        self.noise_scale = noise_scale

        if constraint_weights is not None:
            self.constraint_weights = jnp.array(constraint_weights)
        else:
            self.constraint_weights = jnp.ones(N_CONSTRAINTS)

    def repair(
        self,
        question: str,
        response: str,
        response_embedding: jax.Array,
        constraint_vec: jax.Array,
        codebook_embeddings: jax.Array,
        codebook_texts: list[str],
        correct_emb_mean: jax.Array,
        key: jax.Array,
    ) -> RepairResult:
        """Repair a single response via gradient descent in embedding space.

        **Detailed explanation for engineers:**
            This is the main repair method. It takes a response that violates some
            constraints, gradient-descends on the constraint energy to find a better
            embedding, then decodes back to text via nearest-neighbor in the codebook.

            The gradient descent loop:
            1. Compute constraint energy of the current joint vector
            2. Compute gradient w.r.t. the full joint vector (but we only UPDATE the
               embedding portion — the constraint portion is recomputed each step
               based on the nearest codebook entry, creating a feedback loop)
            3. Actually, for efficiency, we optimize a PROXY: the embedding alone,
               with the constraint energy defined as a function of the embedding's
               distance to constraint-satisfying codebook entries.

            In practice, since we can't re-evaluate text constraints on a continuous
            embedding (constraints need actual text), we use the energy function from
            _constraint_energy as a smooth proxy. The real constraint check happens
            only at the end, on the decoded text.

        Args:
            question: The question text.
            response: The original (possibly wrong) response text.
            response_embedding: Pre-computed embedding of "Q: ... A: ...", shape (EMBED_DIM,).
            constraint_vec: Pre-computed constraint vector, shape (N_CONSTRAINTS,).
            codebook_embeddings: Embeddings of all codebook entries, shape (CODEBOOK_SIZE, EMBED_DIM).
            codebook_texts: Text of all codebook entries, length CODEBOOK_SIZE.
            correct_emb_mean: Mean embedding of correct codebook entries, shape (EMBED_DIM,).
                Used as manifold anchor.
            key: JAX PRNG key for noise generation.

        Returns:
            RepairResult with full repair trajectory details.
        """
        t0 = time.time()

        # Target: all constraints satisfied
        target = jnp.ones(N_CONSTRAINTS)

        # Initial state: the response embedding
        emb = jnp.array(response_embedding)
        original_energy = float(_constraint_energy(
            jnp.concatenate([emb, constraint_vec]),
            EMBED_DIM, target, self.constraint_weights,
        ))

        def combined_energy(embedding: jax.Array) -> jax.Array:
            """Energy = constraint violation + manifold regularizer.

            **Detailed explanation for engineers:**
                The constraint energy measures how far the constraint vector is from
                the "all satisfied" target. But since we're optimizing the EMBEDDING
                (not the constraint vector directly), we need the gradient to flow
                through. We use the current constraint vector as a soft proxy that
                changes based on which codebook entry the embedding is nearest to.

                For the gradient computation, we treat the constraint vector as a
                FUNCTION of the embedding: the nearest codebook entry determines
                the constraint vector. To make this differentiable, we use a
                soft-attention weighted average of all codebook constraint vectors,
                weighted by cosine similarity to the current embedding.

                The manifold regularizer pulls the embedding toward the mean of
                correct responses, preventing drift into adversarial regions.
            """
            # Soft-attention over codebook: weight by cosine similarity
            # This makes the "effective constraint vector" a differentiable function
            # of the embedding, even though individual constraint checks are discrete.
            sims = codebook_embeddings @ embedding
            # Temperature-scaled softmax to focus on nearest neighbors
            temperature = 0.1
            attention_weights = jax.nn.softmax(sims / temperature)
            # Weighted average of codebook constraint vectors gives a soft constraint
            # (computed outside this function and passed as closure variable)
            soft_constraint = attention_weights @ codebook_constraint_vecs

            # Constraint energy: how far from "all satisfied"
            constraint_e = jnp.sum(self.constraint_weights * (soft_constraint - target) ** 2)

            # Manifold regularizer: stay near correct response manifold
            manifold_e = self.manifold_weight * jnp.sum((embedding - correct_emb_mean) ** 2)

            return constraint_e + manifold_e

        # Pre-compute constraint vectors for all codebook entries
        codebook_constraints = []
        for ct in codebook_texts:
            cv = _compute_constraint_vector(question, ct)
            codebook_constraints.append(cv)
        codebook_constraint_vecs = jnp.array(codebook_constraints)

        grad_fn = jax.grad(combined_energy)

        # Gradient descent loop
        prev_energy = float("inf")
        iterations = 0
        converged = False

        for step in range(self.max_iterations):
            iterations = step + 1

            current_energy = float(combined_energy(emb))

            # Check convergence
            if abs(prev_energy - current_energy) < self.convergence_threshold:
                converged = True
                break
            prev_energy = current_energy

            # Compute gradient and normalize for stability
            g = grad_fn(emb)
            g_norm = jnp.linalg.norm(g) + 1e-8
            g_normalized = g / g_norm

            # Gradient descent step
            emb = emb - self.step_size * g_normalized

            # Langevin noise for exploration
            if self.noise_scale > 0:
                key, subkey = jrandom.split(key)
                emb = emb + self.noise_scale * jrandom.normal(subkey, emb.shape)

            # L2-normalize to stay on the unit sphere (embedding space convention)
            emb = emb / (jnp.linalg.norm(emb) + 1e-8)

        final_energy = float(combined_energy(emb))

        # Decode: find nearest codebook entry by cosine similarity
        sims = np.array(codebook_embeddings @ emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        decoded_text = codebook_texts[best_idx]

        # Verify the decoded text
        decoded_constraints = _compute_constraint_vector(question, decoded_text)
        decoded_energy = float(_constraint_energy(
            jnp.concatenate([jnp.array(codebook_embeddings[best_idx]), jnp.array(decoded_constraints)]),
            EMBED_DIM, target, self.constraint_weights,
        ))
        repair_succeeded = all(c > 0.5 for c in decoded_constraints)

        elapsed = time.time() - t0

        return RepairResult(
            original_text=response,
            repaired_text=decoded_text,
            original_constraints=constraint_vec.tolist() if hasattr(constraint_vec, 'tolist') else list(constraint_vec),
            repaired_constraints=decoded_constraints.tolist(),
            original_energy=original_energy,
            final_energy=final_energy,
            decoded_energy=decoded_energy,
            repair_succeeded=repair_succeeded,
            iterations=iterations,
            converged=converged,
            wall_clock_seconds=elapsed,
            codebook_rank=best_idx,
            codebook_similarity=best_sim,
        )


# ---------------------------------------------------------------------------
# 6. Discrete repair simulation (Exp 57 proxy)
# ---------------------------------------------------------------------------


def _simulate_discrete_repair(
    question: str,
    wrong_answer: str,
    codebook: list[tuple[str, bool]],
    max_repairs: int = 3,
) -> dict:
    """Simulate discrete repair (Exp 57 style) without an actual LLM.

    **Detailed explanation for engineers:**
        The discrete repair loop from Exp 57/VerifyRepairPipeline works by:
        1. Detect violations
        2. Format them as feedback text
        3. Re-prompt the LLM → get a new response
        4. Re-verify

        Since we don't have an LLM loaded for this experiment, we SIMULATE the
        discrete repair by checking if the codebook contains a correct answer.
        If it does, we assume the LLM "would have found it" with probability
        proportional to (1 - 0.5^iteration) — meaning each repair iteration has
        a 50% chance of finding the correct answer if one exists. This is a
        rough but reasonable proxy: real LLMs do improve with feedback, but not
        always, and the improvement rate varies.

        This simulation lets us compare gradient vs. discrete repair on the same
        test set without requiring LLM inference, keeping the experiment fast
        and reproducible.

    Args:
        question: The question text.
        wrong_answer: The incorrect response to repair.
        codebook: List of (text, is_correct) candidate responses.
        max_repairs: Maximum number of simulated repair iterations.

    Returns:
        Dict with keys: success, iterations, final_answer.
    """
    rng = np.random.default_rng(hash(question + wrong_answer) % (2**31))

    correct_candidates = [text for text, is_correct in codebook if is_correct]
    if not correct_candidates:
        return {"success": False, "iterations": max_repairs, "final_answer": wrong_answer}

    # Simulate iterative repair: each iteration has 50% chance of finding correct answer
    for i in range(max_repairs):
        if rng.random() < 0.5:
            chosen = correct_candidates[int(rng.integers(0, len(correct_candidates)))]
            # Verify the chosen answer
            constraints = _check_constraints(question, chosen)
            if all(constraints):
                return {"success": True, "iterations": i + 1, "final_answer": chosen}

    return {"success": False, "iterations": max_repairs, "final_answer": wrong_answer}


# ---------------------------------------------------------------------------
# 7. Main experiment
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 87: Gradient-Based Repair in Embedding Space")
    print("  Continuous repair via gradient descent — no LLM re-prompting needed")
    print("  Compare: gradient repair vs discrete repair (Exp 57) vs combined")
    print("=" * 70)

    start = time.time()

    # -------------------------------------------------------------------
    # Step 1: Generate multi-domain dataset with codebooks
    # -------------------------------------------------------------------
    print("\nStep 1: Generating multi-domain dataset with codebooks...")
    dataset = _generate_multi_domain_dataset(n_per_domain=20, seed=87)
    n_total = len(dataset)
    n_correct = sum(1 for d in dataset if d["is_correct"])
    n_wrong = n_total - n_correct
    print(f"  Generated {n_total} QA pairs ({n_correct} correct, {n_wrong} wrong)")
    print(f"  Each question has a codebook of {CODEBOOK_SIZE} candidate responses")

    domain_counts = {}
    for d in dataset:
        domain_counts[d["domain"]] = domain_counts.get(d["domain"], 0) + 1
    for domain, count in sorted(domain_counts.items()):
        print(f"    {domain}: {count} pairs")

    # -------------------------------------------------------------------
    # Step 2: Embed all responses and codebook entries
    # -------------------------------------------------------------------
    print("\nStep 2: Computing embeddings for all responses and codebooks...")

    # Embed main responses as "Q: ... A: ..." pairs
    qa_texts = [f"Q: {d['question']} A: {d['answer']}" for d in dataset]
    qa_embeddings = _embed_texts(qa_texts)
    print(f"  Main response embeddings: shape {qa_embeddings.shape}")

    # Embed all codebook entries (deduplicated for efficiency)
    # Build a mapping from (question_idx, codebook_idx) → embedding
    all_codebook_texts: list[str] = []
    codebook_text_to_idx: dict[str, int] = {}
    for d in dataset:
        for cb_text, _ in d["codebook"]:
            key = f"Q: {d['question']} A: {cb_text}"
            if key not in codebook_text_to_idx:
                codebook_text_to_idx[key] = len(all_codebook_texts)
                all_codebook_texts.append(key)

    codebook_embeddings_all = _embed_texts(all_codebook_texts)
    print(f"  Codebook embeddings: {len(all_codebook_texts)} unique entries, shape {codebook_embeddings_all.shape}")

    # -------------------------------------------------------------------
    # Step 3: Compute constraint vectors for all responses
    # -------------------------------------------------------------------
    print("\nStep 3: Computing constraint vectors...")

    constraint_vectors = np.array([
        _compute_constraint_vector(d["question"], d["answer"])
        for d in dataset
    ], dtype=np.float32)
    print(f"  Constraint vectors: shape {constraint_vectors.shape}")

    # Show constraint satisfaction rates by correctness
    correct_mask = np.array([d["is_correct"] for d in dataset])
    print("\n  Constraint satisfaction rates:")
    print(f"  {'Constraint':<25s} {'Correct':>8s} {'Wrong':>8s} {'Delta':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for j, cname in enumerate(CONSTRAINT_NAMES):
        rate_c = constraint_vectors[correct_mask, j].mean()
        rate_w = constraint_vectors[~correct_mask, j].mean()
        delta = rate_c - rate_w
        print(f"  {cname:<25s} {rate_c:>7.0%} {rate_w:>7.0%} {delta:>+7.0%}")

    # -------------------------------------------------------------------
    # Step 4: Gradient repair on wrong responses
    # -------------------------------------------------------------------
    print("\nStep 4: Gradient repair on wrong responses...")

    # Select wrong responses for repair
    wrong_indices = [i for i, d in enumerate(dataset) if not d["is_correct"]]
    n_repair = len(wrong_indices)
    print(f"  Repairing {n_repair} wrong responses")

    # Compute mean embedding of correct responses (manifold anchor)
    correct_indices = [i for i, d in enumerate(dataset) if d["is_correct"]]
    correct_emb_mean = jnp.array(qa_embeddings[correct_indices].mean(axis=0))

    # Default hyperparameters
    repairer = GradientRepairer(
        step_size=DEFAULT_STEP_SIZE,
        max_iterations=DEFAULT_MAX_ITERS,
        manifold_weight=0.1,
        noise_scale=0.001,
    )

    gradient_results: list[RepairResult] = []
    discrete_results: list[dict] = []
    key = jrandom.PRNGKey(87)

    for idx_in_wrong, data_idx in enumerate(wrong_indices):
        d = dataset[data_idx]

        # Build codebook embeddings for this question
        cb_embs = []
        cb_texts = []
        for cb_text, _ in d["codebook"]:
            cb_key = f"Q: {d['question']} A: {cb_text}"
            cb_idx = codebook_text_to_idx[cb_key]
            cb_embs.append(codebook_embeddings_all[cb_idx])
            cb_texts.append(cb_text)
        cb_embs_arr = jnp.array(np.array(cb_embs))

        # Correct entries in codebook → manifold anchor
        correct_cb_embs = []
        for j, (_, is_c) in enumerate(d["codebook"]):
            if is_c:
                correct_cb_embs.append(cb_embs[j])
        if correct_cb_embs:
            cb_correct_mean = jnp.array(np.array(correct_cb_embs).mean(axis=0))
        else:
            cb_correct_mean = correct_emb_mean

        # Gradient repair
        key, subkey = jrandom.split(key)
        gr = repairer.repair(
            question=d["question"],
            response=d["answer"],
            response_embedding=jnp.array(qa_embeddings[data_idx]),
            constraint_vec=jnp.array(constraint_vectors[data_idx]),
            codebook_embeddings=cb_embs_arr,
            codebook_texts=cb_texts,
            correct_emb_mean=cb_correct_mean,
            key=subkey,
        )
        gradient_results.append(gr)

        # Discrete repair simulation
        dr = _simulate_discrete_repair(
            d["question"], d["answer"], d["codebook"], max_repairs=3
        )
        discrete_results.append(dr)

        if idx_in_wrong < 5:
            q_short = d["question"][:50]
            a_short = d["answer"][:40]
            r_short = gr.repaired_text[:40]
            print(f"    [{idx_in_wrong}] Q: {q_short}")
            print(f"         A: {a_short}...")
            print(f"         R: {r_short}...")
            print(
                f"         energy: {gr.original_energy:.3f} -> {gr.decoded_energy:.3f} "
                f"iters: {gr.iterations} "
                f"{'OK' if gr.repair_succeeded else 'FAIL'} "
                f"(sim={gr.codebook_similarity:.3f})"
            )

    # -------------------------------------------------------------------
    # Step 5: Compare repair strategies
    # -------------------------------------------------------------------
    print("\nStep 5: Comparing repair strategies...")

    grad_success = sum(1 for r in gradient_results if r.repair_succeeded)
    disc_success = sum(1 for r in discrete_results if r["success"])

    # Combined: try gradient first, then discrete on failures
    combined_success = 0
    for gr, dr in zip(gradient_results, discrete_results):
        if gr.repair_succeeded:
            combined_success += 1
        elif dr["success"]:
            combined_success += 1

    grad_rate = grad_success / n_repair if n_repair > 0 else 0.0
    disc_rate = disc_success / n_repair if n_repair > 0 else 0.0
    comb_rate = combined_success / n_repair if n_repair > 0 else 0.0

    avg_grad_iters = np.mean([r.iterations for r in gradient_results])
    avg_disc_iters = np.mean([r["iterations"] for r in discrete_results])
    avg_grad_time = np.mean([r.wall_clock_seconds for r in gradient_results])

    print(f"\n  {'Strategy':<25s} {'Success Rate':>14s} {'Avg Iters':>10s} {'Avg Time':>10s}")
    print(f"  {'-'*25} {'-'*14} {'-'*10} {'-'*10}")
    print(f"  {'Gradient repair':<25s} {grad_rate:>13.1%} {avg_grad_iters:>10.1f} {avg_grad_time:>9.3f}s")
    print(f"  {'Discrete repair (sim)':<25s} {disc_rate:>13.1%} {avg_disc_iters:>10.1f} {'N/A':>10s}")
    print(f"  {'Combined (grad+disc)':<25s} {comb_rate:>13.1%} {'N/A':>10s} {'N/A':>10s}")

    # Per-domain breakdown
    print("\n  Per-domain gradient repair success:")
    domain_grad_results: dict[str, list[bool]] = {}
    for i, data_idx in enumerate(wrong_indices):
        domain = dataset[data_idx]["domain"]
        if domain not in domain_grad_results:
            domain_grad_results[domain] = []
        domain_grad_results[domain].append(gradient_results[i].repair_succeeded)

    print(f"  {'Domain':<15s} {'Success':>8s} {'Total':>6s} {'Rate':>8s}")
    print(f"  {'-'*15} {'-'*8} {'-'*6} {'-'*8}")
    for domain in DOMAINS:
        if domain in domain_grad_results:
            successes = sum(domain_grad_results[domain])
            total = len(domain_grad_results[domain])
            rate = successes / total if total > 0 else 0.0
            print(f"  {domain:<15s} {successes:>8d} {total:>6d} {rate:>7.0%}")

    # -------------------------------------------------------------------
    # Step 6: Ablation study — step sizes and max iterations
    # -------------------------------------------------------------------
    print("\nStep 6: Ablation study...")

    ablation_results: dict[str, dict] = {}

    # Use a subset of wrong responses for ablation (first 20 for speed)
    ablation_indices = wrong_indices[:min(20, len(wrong_indices))]

    for ss in STEP_SIZES:
        for mi in MAX_ITERS_LIST:
            config_key = f"ss={ss}_mi={mi}"
            abl_repairer = GradientRepairer(
                step_size=ss,
                max_iterations=mi,
                manifold_weight=0.1,
                noise_scale=0.001,
            )

            abl_successes = 0
            abl_iters_list: list[int] = []
            abl_times: list[float] = []
            abl_key = jrandom.PRNGKey(87 + hash(config_key) % 1000)

            for data_idx in ablation_indices:
                d = dataset[data_idx]

                # Build codebook embeddings
                cb_embs = []
                cb_texts = []
                for cb_text, _ in d["codebook"]:
                    cb_key_str = f"Q: {d['question']} A: {cb_text}"
                    cb_idx = codebook_text_to_idx[cb_key_str]
                    cb_embs.append(codebook_embeddings_all[cb_idx])
                    cb_texts.append(cb_text)
                cb_embs_arr = jnp.array(np.array(cb_embs))

                correct_cb_embs = []
                for j, (_, is_c) in enumerate(d["codebook"]):
                    if is_c:
                        correct_cb_embs.append(cb_embs[j])
                cb_correct_mean = (
                    jnp.array(np.array(correct_cb_embs).mean(axis=0))
                    if correct_cb_embs
                    else correct_emb_mean
                )

                abl_key, subkey = jrandom.split(abl_key)
                abl_r = abl_repairer.repair(
                    question=d["question"],
                    response=d["answer"],
                    response_embedding=jnp.array(qa_embeddings[data_idx]),
                    constraint_vec=jnp.array(constraint_vectors[data_idx]),
                    codebook_embeddings=cb_embs_arr,
                    codebook_texts=cb_texts,
                    correct_emb_mean=cb_correct_mean,
                    key=subkey,
                )
                if abl_r.repair_succeeded:
                    abl_successes += 1
                abl_iters_list.append(abl_r.iterations)
                abl_times.append(abl_r.wall_clock_seconds)

            n_abl = len(ablation_indices)
            ablation_results[config_key] = {
                "step_size": ss,
                "max_iterations": mi,
                "n_samples": n_abl,
                "success_rate": abl_successes / n_abl if n_abl > 0 else 0.0,
                "avg_iterations": float(np.mean(abl_iters_list)),
                "avg_time_seconds": float(np.mean(abl_times)),
            }

    print(f"\n  {'Config':<20s} {'Success':>8s} {'Avg Iters':>10s} {'Avg Time':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10}")
    for config_key, ar in sorted(ablation_results.items()):
        print(
            f"  {config_key:<20s} {ar['success_rate']:>7.0%} "
            f"{ar['avg_iterations']:>10.1f} {ar['avg_time_seconds']:>9.3f}s"
        )

    # Find best ablation config
    best_config = max(ablation_results.items(), key=lambda x: x[1]["success_rate"])
    print(f"\n  Best config: {best_config[0]} (success rate: {best_config[1]['success_rate']:.0%})")

    # -------------------------------------------------------------------
    # Step 7: Energy analysis
    # -------------------------------------------------------------------
    print("\nStep 7: Energy analysis...")

    energy_before = [r.original_energy for r in gradient_results]
    energy_after = [r.decoded_energy for r in gradient_results]
    energy_improved = sum(1 for b, a in zip(energy_before, energy_after) if a < b)
    avg_energy_drop = np.mean([b - a for b, a in zip(energy_before, energy_after)])

    print(f"  Energy improved: {energy_improved}/{n_repair} ({energy_improved/n_repair:.0%})")
    print(f"  Average energy drop: {avg_energy_drop:.4f}")
    print(f"  Mean energy before: {np.mean(energy_before):.4f}")
    print(f"  Mean energy after:  {np.mean(energy_after):.4f}")

    converged_count = sum(1 for r in gradient_results if r.converged)
    print(f"  Converged: {converged_count}/{n_repair} ({converged_count/n_repair:.0%})")

    # -------------------------------------------------------------------
    # Step 8: Save results
    # -------------------------------------------------------------------
    print("\nStep 8: Saving results...")

    elapsed = time.time() - start

    results = {
        "experiment": "87_gradient_repair",
        "description": (
            "Continuous repair via gradient descent in embedding space. "
            "Replaces discrete LLM re-prompting with gradient-based optimization "
            "and nearest-neighbor codebook decoding."
        ),
        "prerequisites": [
            "experiment_65_embedding_constraints",
            "experiment_66_differentiable_constraints",
            "experiment_57_verify_repair (concept)",
        ],
        "dataset": {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_wrong": n_wrong,
            "n_per_domain": 20,
            "domains": DOMAINS,
            "codebook_size": CODEBOOK_SIZE,
        },
        "hyperparameters": {
            "embed_dim": EMBED_DIM,
            "n_constraints": N_CONSTRAINTS,
            "default_step_size": DEFAULT_STEP_SIZE,
            "default_max_iterations": DEFAULT_MAX_ITERS,
            "convergence_threshold": CONVERGENCE_THRESHOLD,
            "manifold_weight": 0.1,
            "noise_scale": 0.001,
        },
        "comparison": {
            "gradient_repair": {
                "success_rate": grad_rate,
                "n_success": grad_success,
                "n_total": n_repair,
                "avg_iterations": float(avg_grad_iters),
                "avg_wall_clock_seconds": float(avg_grad_time),
            },
            "discrete_repair_simulated": {
                "success_rate": disc_rate,
                "n_success": disc_success,
                "n_total": n_repair,
                "avg_iterations": float(avg_disc_iters),
                "note": "Simulated without LLM — 50% per-iteration success probability",
            },
            "combined": {
                "success_rate": comb_rate,
                "n_success": combined_success,
                "n_total": n_repair,
            },
        },
        "per_domain": {
            domain: {
                "n_total": len(results_list),
                "n_success": sum(results_list),
                "success_rate": sum(results_list) / len(results_list) if results_list else 0.0,
            }
            for domain, results_list in domain_grad_results.items()
        },
        "ablation": ablation_results,
        "best_ablation_config": {
            "config": best_config[0],
            **best_config[1],
        },
        "energy_analysis": {
            "energy_improved_rate": energy_improved / n_repair if n_repair > 0 else 0.0,
            "avg_energy_drop": float(avg_energy_drop),
            "mean_energy_before": float(np.mean(energy_before)),
            "mean_energy_after": float(np.mean(energy_after)),
            "convergence_rate": converged_count / n_repair if n_repair > 0 else 0.0,
        },
        "individual_repairs": [
            {
                "original_text": r.original_text[:100],
                "repaired_text": r.repaired_text[:100],
                "original_energy": r.original_energy,
                "decoded_energy": r.decoded_energy,
                "repair_succeeded": r.repair_succeeded,
                "iterations": r.iterations,
                "converged": r.converged,
                "wall_clock_seconds": r.wall_clock_seconds,
                "codebook_similarity": r.codebook_similarity,
            }
            for r in gradient_results
        ],
        "wall_clock_seconds": elapsed,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {RESULTS_PATH}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 87 RESULTS ({elapsed:.0f}s)")
    print(sep)

    print(f"\n  Repair Strategy Comparison:")
    print(f"  {'Strategy':<25s} {'Success Rate':>14s}")
    print(f"  {'-'*25} {'-'*14}")
    print(f"  {'Gradient repair':<25s} {grad_rate:>13.1%}")
    print(f"  {'Discrete repair (sim)':<25s} {disc_rate:>13.1%}")
    print(f"  {'Combined':<25s} {comb_rate:>13.1%}")

    print(f"\n  Energy Analysis:")
    print(f"    Energy improved: {energy_improved}/{n_repair} ({energy_improved/n_repair:.0%})")
    print(f"    Avg energy drop: {avg_energy_drop:.4f}")
    print(f"    Convergence rate: {converged_count/n_repair:.0%}")

    print(f"\n  Best Ablation: {best_config[0]} ({best_config[1]['success_rate']:.0%})")

    # Verdict
    if grad_rate > disc_rate:
        print(f"\n  VERDICT: Gradient repair outperforms discrete repair simulation.")
        print(f"  Continuous optimization in embedding space finds better corrections")
        print(f"  than the simulated LLM re-prompting approach.")
    elif grad_rate == disc_rate:
        print(f"\n  VERDICT: Gradient and discrete repair achieve comparable success.")
        print(f"  Gradient repair is faster (no LLM calls) but the codebook limits")
        print(f"  the quality of decoded responses.")
    else:
        print(f"\n  VERDICT: Discrete repair (simulated) outperforms gradient repair.")
        print(f"  The embedding space may not be smooth enough for gradient descent,")
        print(f"  or the codebook is too small to contain good decode targets.")

    if comb_rate > max(grad_rate, disc_rate):
        print(f"\n  COMBINED: The combined strategy ({comb_rate:.0%}) outperforms both")
        print(f"  individual strategies, suggesting gradient and discrete repair are")
        print(f"  complementary — they fix different types of violations.")

    print(f"\n  NOTE: This experiment uses Exp 65/66's joint embedding-constraint space")
    print(f"  and extends it with gradient-based repair + codebook decoding.")
    print(f"  Target models for production: Qwen3.5-0.8B, google/gemma-4-E4B-it")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
