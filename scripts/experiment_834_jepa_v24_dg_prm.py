#!/usr/bin/env python3
"""Experiment 834: JEPA v24 with DG-PRM domain-balanced corpus and domain reweighting.

**Researcher summary:**
    JEPA v23 (Exp 824) collapsed on ARC-Challenge (AUC=0.04) because the LIMO
    corpus contained zero ARC training examples.  This experiment rebuilds JEPA
    using a balanced corpus (GSM8K + HumanEval + ARC + SVAMP) and applies three
    domain-reweighting techniques from recent arXiv literature:

    1. DG-PRM domain head (arXiv 2507.17849): 4-class softmax domain classifier
       trained jointly with the binary correctness head. At inference, the
       correctness score is multiplied by a domain weight based on which domain
       the model thinks the input belongs to.

    2. DreamPRM per-domain loss weighting (arXiv 2505.20241): the training loss
       for each sample is scaled by the inverse of that domain's validation
       performance. Domains that performed poorly (ARC) receive higher loss
       weights, forcing the model to focus on them.

    3. ΔEnergy triplet loss (arXiv 2510.11296): for each (anchor, positive,
       negative) triplet, the triplet loss is weighted by the magnitude of the
       energy gap between positive and negative.  Triplets with a large energy
       gap provide a clearer learning signal and receive proportionally higher
       weight.

**Why ARC failed before:**
    The LIMO corpus (Exp 824) selected pairs from GSM8K (50 pairs), HumanEval
    (10 pairs), and SVAMP (10 pairs).  No ARC examples were included.  When the
    model was then evaluated on ARC reasoning steps, the embedding space had
    never been trained to distinguish correct from incorrect ARC-style deductions.
    The result was AUC=0.04 — worse than random (0.50) — meaning the model
    actively inverted scores for ARC.  This is a classic out-of-distribution
    failure caused by corpus imbalance.

**Target:**
    All per-domain AUCs > 0.55 (jepa_v24_domain_balanced verdict).

Spec: REQ-LEARN-047, REQ-LEARN-834-001, SCENARIO-LEARN-059, SCENARIO-LEARN-834-001
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without pip install
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 834
TITLE = "JEPA v24: DG-PRM Domain-Balanced Corpus with Domain Reweighting"
DELIVERABLE = "results/experiment_834_jepa_v24_dg_prm.json"

EMBED_DIM = 256
"""Embedding dimension — matches the RandomProjection used by prior JEPA experiments."""

HIDDEN1 = 64
HIDDEN2 = 32

N_CORRECTNESS = 1
"""Binary correctness head output dimension."""

DOMAIN_NAMES = ["gsm8k", "humaneval", "arc", "svamp"]
N_DOMAINS = len(DOMAIN_NAMES)
"""Number of domains for the DG-PRM domain classifier head."""

# DreamPRM per-domain loss weights (arXiv 2505.20241).
# ARC historically gets the highest loss weight because it had the worst
# validation performance across prior JEPA runs (AUC=0.04 in Exp 832/824).
DREAM_PRM_WEIGHTS: dict[str, float] = {
    "gsm8k": 1.0,
    "humaneval": 1.5,
    "arc": 5.0,
    "svamp": 1.5,
}

# DG-PRM inference domain weights (arXiv 2507.17849).
# These multiplicative factors scale the final correctness score at inference
# time.  ARC gets 3x to compensate for the model's historical under-scoring.
DG_PRM_DOMAIN_WEIGHTS: dict[str, float] = {
    "gsm8k": 1.0,
    "humaneval": 1.5,
    "arc": 3.0,
    "svamp": 1.5,
}

# ΔEnergy triplet loss weight clamping range (arXiv 2510.11296).
# Clamping prevents degenerate cases where a huge energy gap swamps all
# other triplets, causing training to collapse onto a single pair.
DELTA_ENERGY_MIN = 0.5
DELTA_ENERGY_MAX = 3.0

N_EPOCHS = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
VAL_FRACTION = 0.2
TRIPLET_MARGIN = 0.5

# ---------------------------------------------------------------------------
# Corpus: balanced synthetic pairs
# ---------------------------------------------------------------------------

# Five distinct syllogism patterns for ARC variety.
#   Pattern 1: A→B, B→C ⊢ A→C (standard transitivity)
#   Pattern 2: All A are B, X is A ⊢ X is B (universal instantiation)
#   Pattern 3: If P then Q, P ⊢ Q (modus ponens)
#   Pattern 4: A→B, ¬B ⊢ ¬A (modus tollens)
#   Pattern 5: All X have Y, Z has Y ⊢ Z is X (generalisation — INVALID as correct;
#              only the VALID deduction form is in the correct set)
#
# Incorrect examples use the invalid converse/inverse forms that look plausible
# but are logically unsound.

ARC_CORRECT_STEPS = [
    # Pattern 1 — transitivity (×4)
    "If all mammals are warm-blooded and all warm-blooded animals regulate temperature, then all mammals regulate temperature.",
    "If plants need sunlight to grow and sunlight requires clear skies, then plants require clear skies to grow.",
    "If iron conducts electricity and electricity causes heating, then iron causes heating when current flows.",
    "If viruses infect cells and cells are part of organisms, then viruses affect organisms.",
    # Pattern 2 — universal instantiation (×4)
    "All reptiles are cold-blooded. A lizard is a reptile. Therefore, a lizard is cold-blooded.",
    "All prime numbers greater than 2 are odd. 7 is a prime greater than 2. Therefore, 7 is odd.",
    "All acids have pH < 7. Hydrochloric acid is an acid. Therefore, hydrochloric acid has pH < 7.",
    "All gases expand when heated. Oxygen is a gas. Therefore, oxygen expands when heated.",
    # Pattern 3 — modus ponens (×4)
    "If it rains, the ground gets wet. It is raining. Therefore, the ground is wet.",
    "If a shape has four equal sides and four right angles, it is a square. This shape has four equal sides and four right angles. Therefore, it is a square.",
    "If temperature drops below 0°C, water freezes. Temperature is −5°C. Therefore, water is frozen.",
    "If a number is divisible by 4 it is even. 16 is divisible by 4. Therefore, 16 is even.",
    # Pattern 4 — modus tollens (×4)
    "If the car has fuel, it can start. The car cannot start. Therefore, the car has no fuel.",
    "If a cell has a nucleus, it is eukaryotic. This cell is not eukaryotic. Therefore, it has no nucleus.",
    "If the experiment is controlled, results are reliable. Results are unreliable. Therefore, the experiment is not controlled.",
    "If all sides of a triangle are equal, it is equilateral. This triangle is not equilateral. Therefore, not all sides are equal.",
    # Pattern 5 — valid form: specific instance follows from universal rule (×4)
    "All metals conduct heat. Copper is a metal. We can conclude copper conducts heat.",
    "All planets orbit a star. Earth is a planet. We can conclude Earth orbits a star.",
    "All chemical reactions obey conservation of mass. Combustion is a chemical reaction. Therefore, combustion obeys conservation of mass.",
    "All living organisms require energy. A bacterium is a living organism. Therefore, a bacterium requires energy.",
]

ARC_INCORRECT_STEPS = [
    # Converse errors — A→B does NOT mean B→A
    "If all mammals are warm-blooded, then all warm-blooded animals must be mammals.",
    "If plants need sunlight, then sunlight exists only to serve plants.",
    "If iron conducts electricity, then anything that heats must be iron.",
    "If viruses infect cells, then all cells must contain viruses.",
    # Affirming the consequent — incorrect form of universal instantiation
    "All reptiles are cold-blooded. A lizard is cold-blooded. Therefore, a lizard must be a reptile.",
    "All prime numbers greater than 2 are odd. 9 is odd. Therefore, 9 must be prime.",
    "All acids have pH < 7. A substance has pH < 7. Therefore, the substance must be an acid.",
    "All gases expand when heated. Oxygen expands. Therefore, oxygen must have been heated.",
    # Denying the antecedent — if P→Q, ¬P does NOT imply ¬Q
    "If it rains, the ground gets wet. It is not raining. Therefore, the ground cannot be wet.",
    "If a shape has four equal sides and four right angles, it is a square. This shape is not a square. Therefore, it cannot have four equal sides.",
    "If temperature drops below 0°C, water freezes. Temperature is 10°C. Therefore, water cannot freeze.",
    "If a number is divisible by 4 it is even. 6 is not divisible by 4. Therefore, 6 is not even.",
    # Inverse fallacy — if P→Q, ¬P→¬Q is invalid
    "If the car has fuel, it can start. The car has fuel. Therefore, it will definitely start under all conditions.",
    "If a cell has a nucleus, it is eukaryotic. This cell is eukaryotic. Therefore, it must have a nucleus.",
    "If the experiment is controlled, results are reliable. The experiment is controlled. Therefore, no result can ever be unreliable.",
    "If all sides of a triangle are equal, it is equilateral. This triangle has some equal sides. Therefore, it is equilateral.",
    # Generalisation fallacy — observed instance implies universal rule
    "All metals conduct heat. Copper conducts heat. Therefore, copper must be a metal and the only substance conducting heat.",
    "All planets orbit a star. Earth orbits the sun. Therefore, anything that orbits must be a planet.",
    "Combustion obeys conservation of mass. Therefore, all mass-conserving processes must be combustion.",
    "A bacterium requires energy. Therefore, all energy-requiring things must be bacteria.",
]

assert len(ARC_CORRECT_STEPS) == 20, "Need exactly 20 ARC correct steps"
assert len(ARC_INCORRECT_STEPS) == 20, "Need exactly 20 ARC incorrect steps"

GSM8K_CORRECT_STEPS = [
    "To find the total cost: 3 items × $4 each = $12 total. So the answer is $12.",
    "Mary has 15 apples and gives away 7. Remaining = 15 − 7 = 8 apples.",
    "A recipe needs 2 cups of flour per batch. For 5 batches: 2 × 5 = 10 cups.",
    "Train travels 60 mph for 3 hours. Distance = 60 × 3 = 180 miles.",
    "Store sells 24 items in 4 days equally. Per day = 24 ÷ 4 = 6 items.",
    "Tom saves $5 per week for 8 weeks. Total savings = 5 × 8 = $40.",
    "A box holds 12 oranges. 5 boxes contain 12 × 5 = 60 oranges total.",
    "Temperature drops 3°C per hour for 4 hours. Total drop = 3 × 4 = 12°C.",
    "Class has 30 students, 12 absent. Present = 30 − 12 = 18 students.",
    "Pipe fills 1/6 of a tank per hour. In 6 hours it fills 6 × (1/6) = 1 full tank.",
    "If 4 workers finish in 6 days, 1 worker takes 4 × 6 = 24 days.",
    "Profit = revenue − cost = $150 − $90 = $60.",
    "Area of rectangle = length × width = 8 × 5 = 40 square units.",
    "Speed = distance ÷ time = 120 km ÷ 2 h = 60 km/h.",
    "Discount = 20% of $80 = 0.20 × 80 = $16. Sale price = $80 − $16 = $64.",
    "Perimeter of square = 4 × side = 4 × 7 = 28 units.",
    "Average of 4 numbers: (10 + 20 + 30 + 40) ÷ 4 = 100 ÷ 4 = 25.",
    "Population grows by 5% of 1000 = 50. New total = 1000 + 50 = 1050.",
    "10 pens cost $30 total. Per pen = $30 ÷ 10 = $3 each.",
    "A number doubled and then increased by 3 equals 11. Original = (11 − 3) ÷ 2 = 4.",
]

GSM8K_INCORRECT_STEPS = [
    "To find the total cost: 3 items × $4 each = $8 total. So the answer is $8.",
    "Mary has 15 apples and gives away 7. Remaining = 15 + 7 = 22 apples.",
    "A recipe needs 2 cups of flour per batch. For 5 batches: 2 + 5 = 7 cups.",
    "Train travels 60 mph for 3 hours. Distance = 60 + 3 = 63 miles.",
    "Store sells 24 items in 4 days equally. Per day = 24 × 4 = 96 items.",
    "Tom saves $5 per week for 8 weeks. Total savings = 5 + 8 = $13.",
    "A box holds 12 oranges. 5 boxes contain 12 + 5 = 17 oranges total.",
    "Temperature drops 3°C per hour for 4 hours. Total drop = 3 + 4 = 7°C.",
    "Class has 30 students, 12 absent. Present = 30 × 12 = 360 students.",
    "Pipe fills 1/6 of a tank per hour. In 6 hours it fills 6 + (1/6) ≈ 6.17 tanks.",
    "If 4 workers finish in 6 days, 1 worker takes 6 ÷ 4 = 1.5 days.",
    "Profit = revenue × cost = $150 × $90 = $13500.",
    "Area of rectangle = length + width = 8 + 5 = 13 square units.",
    "Speed = distance × time = 120 km × 2 h = 240 km/h.",
    "Discount = 20% of $80 = 0.20 + 80 = $80.20. Sale price = $80 − $80.20 = −$0.20.",
    "Perimeter of square = 2 × side = 2 × 7 = 14 units.",
    "Average of 4 numbers: (10 + 20 + 30 + 40) × 4 = 400.",
    "Population grows by 5% of 1000 = 50. New total = 1000 − 50 = 950.",
    "10 pens cost $30 total. Per pen = $30 × 10 = $300 each.",
    "A number doubled and then increased by 3 equals 11. Original = (11 + 3) × 2 = 28.",
]

assert len(GSM8K_CORRECT_STEPS) == 20
assert len(GSM8K_INCORRECT_STEPS) == 20

HUMANEVAL_CORRECT_STEPS = [
    "def add(a, b): return a + b  # Correct: returns sum of two numbers.",
    "def is_even(n): return n % 2 == 0  # Correct: True when n divisible by 2.",
    "def max_of_two(a, b): return a if a > b else b  # Correct: returns larger value.",
    "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)  # Correct: recursive factorial.",
    "def reverse_string(s): return s[::-1]  # Correct: reverses string with slice.",
    "def count_vowels(s): return sum(1 for c in s.lower() if c in 'aeiou')  # Correct: counts vowels.",
    "def is_palindrome(s): return s == s[::-1]  # Correct: True when s reads same forwards/backwards.",
    "def sum_list(lst): return sum(lst)  # Correct: sum of all list elements.",
    "def first_element(lst): return lst[0] if lst else None  # Correct: returns first or None.",
    "def square(n): return n * n  # Correct: returns n squared.",
    "def absolute_value(n): return n if n >= 0 else -n  # Correct: returns non-negative magnitude.",
    "def clamp(v, lo, hi): return max(lo, min(hi, v))  # Correct: clamps v to [lo, hi].",
    "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))  # Correct: primality test.",
    "def flatten(lst): return [x for sub in lst for x in sub]  # Correct: flattens one level.",
    "def unique(lst): return list(set(lst))  # Correct: removes duplicates (order not preserved).",
    "def zip_sum(a, b): return [x + y for x, y in zip(a, b)]  # Correct: element-wise sum.",
    "def repeat_str(s, n): return s * n  # Correct: repeats string n times.",
    "def is_sorted(lst): return lst == sorted(lst)  # Correct: True when list is in ascending order.",
    "def count_occurrences(lst, x): return lst.count(x)  # Correct: count of x in lst.",
    "def nth_element(lst, n): return lst[n]  # Correct: returns element at index n.",
]

HUMANEVAL_INCORRECT_STEPS = [
    "def add(a, b): return a - b  # Incorrect: subtracts instead of adding.",
    "def is_even(n): return n % 2 == 1  # Incorrect: True when n is odd, not even.",
    "def max_of_two(a, b): return a if a < b else b  # Incorrect: returns smaller value.",
    "def factorial(n): return n * factorial(n+1)  # Incorrect: infinite recursion, never terminates.",
    "def reverse_string(s): return s  # Incorrect: returns original string unchanged.",
    "def count_vowels(s): return len(s)  # Incorrect: counts all characters, not just vowels.",
    "def is_palindrome(s): return s == s  # Incorrect: always True, not a palindrome check.",
    "def sum_list(lst): return len(lst)  # Incorrect: returns length, not sum.",
    "def first_element(lst): return lst[-1] if lst else None  # Incorrect: returns last element.",
    "def square(n): return n + n  # Incorrect: returns 2n, not n².",
    "def absolute_value(n): return n  # Incorrect: negative numbers remain negative.",
    "def clamp(v, lo, hi): return min(lo, max(hi, v))  # Incorrect: logic is inverted.",
    "def is_prime(n): return n % 2 != 0  # Incorrect: only checks divisibility by 2.",
    "def flatten(lst): return lst  # Incorrect: does not flatten nested lists.",
    "def unique(lst): return lst  # Incorrect: does not remove duplicates.",
    "def zip_sum(a, b): return [x * y for x, y in zip(a, b)]  # Incorrect: multiplies instead of adds.",
    "def repeat_str(s, n): return s + str(n)  # Incorrect: appends n as a string.",
    "def is_sorted(lst): return lst == sorted(lst, reverse=True)  # Incorrect: checks descending order.",
    "def count_occurrences(lst, x): return len(lst)  # Incorrect: returns total length, not count of x.",
    "def nth_element(lst, n): return lst[0]  # Incorrect: always returns first element.",
]

assert len(HUMANEVAL_CORRECT_STEPS) == 20
assert len(HUMANEVAL_INCORRECT_STEPS) == 20

SVAMP_CORRECT_STEPS = [
    "John has 8 marbles. He finds 5 more. Now he has 8 + 5 = 13 marbles.",
    "A baker bakes 36 rolls. Sells 24. Remaining = 36 − 24 = 12 rolls.",
    "Bus seats 48 passengers. 3 trips. Total = 48 × 3 = 144 passengers.",
    "A garden has 5 rows of 9 plants each. Total = 5 × 9 = 45 plants.",
    "A jar holds 500 mL. Fill 3 jars. Total liquid = 500 × 3 = 1500 mL.",
    "Sam earns $12/hour. Works 7 hours. Earns = 12 × 7 = $84.",
    "Box weighs 2 kg. 6 boxes weigh 2 × 6 = 12 kg total.",
    "A team scores 45 points total across 9 games equally. Per game = 45 ÷ 9 = 5 points.",
    "Library has 200 books. Returns 35 more. Now has 200 + 35 = 235 books.",
    "Ribbon of 60 cm is cut into 4 equal pieces. Each = 60 ÷ 4 = 15 cm.",
]

SVAMP_INCORRECT_STEPS = [
    "John has 8 marbles. He finds 5 more. Now he has 8 − 5 = 3 marbles.",
    "A baker bakes 36 rolls. Sells 24. Remaining = 36 + 24 = 60 rolls.",
    "Bus seats 48 passengers. 3 trips. Total = 48 + 3 = 51 passengers.",
    "A garden has 5 rows of 9 plants each. Total = 5 + 9 = 14 plants.",
    "A jar holds 500 mL. Fill 3 jars. Total liquid = 500 + 3 = 503 mL.",
    "Sam earns $12/hour. Works 7 hours. Earns = 12 + 7 = $19.",
    "Box weighs 2 kg. 6 boxes weigh 2 + 6 = 8 kg total.",
    "A team scores 45 points total across 9 games equally. Per game = 45 × 9 = 405 points.",
    "Library has 200 books. Returns 35 more. Now has 200 − 35 = 165 books.",
    "Ribbon of 60 cm is cut into 4 equal pieces. Each = 60 × 4 = 240 cm.",
]

assert len(SVAMP_CORRECT_STEPS) == 10
assert len(SVAMP_INCORRECT_STEPS) == 10


def build_balanced_corpus() -> list[dict[str, Any]]:
    """Build the domain-balanced corpus for JEPA v24 training.

    **Why balanced:**
        JEPA v23 collapsed on ARC because its corpus (LIMO) had 0 ARC examples.
        This corpus guarantees coverage of all four domains before training begins.

    Returns:
        List of dicts with keys: text (str), label (int 0=incorrect/1=correct),
        domain (str), domain_idx (int).
    """
    n_arc_pairs = len(ARC_CORRECT_STEPS)
    assert n_arc_pairs >= 10, (
        f"ARC corpus must have at least 10 pairs, got {n_arc_pairs}. "
        "Refusing to train JEPA v24 without ARC coverage — this was the v23 root cause."
    )

    pairs: list[dict[str, Any]] = []

    # Each domain contributes correct (label=1) and incorrect (label=0) examples.
    domain_data = [
        ("gsm8k", GSM8K_CORRECT_STEPS, GSM8K_INCORRECT_STEPS),
        ("humaneval", HUMANEVAL_CORRECT_STEPS, HUMANEVAL_INCORRECT_STEPS),
        ("arc", ARC_CORRECT_STEPS, ARC_INCORRECT_STEPS),
        ("svamp", SVAMP_CORRECT_STEPS, SVAMP_INCORRECT_STEPS),
    ]

    for domain, corrects, incorrects in domain_data:
        d_idx = DOMAIN_NAMES.index(domain)
        for text in corrects:
            pairs.append({"text": text, "label": 1, "domain": domain, "domain_idx": d_idx})
        for text in incorrects:
            pairs.append({"text": text, "label": 0, "domain": domain, "domain_idx": d_idx})

    return pairs


# ---------------------------------------------------------------------------
# Simple text → embedding (RandomProjection approximation)
# ---------------------------------------------------------------------------

def _embed_text(text: str, dim: int = EMBED_DIM, seed: int = 42) -> np.ndarray:
    """Convert text to a fixed-dimensional embedding using a hash-based projection.

    **Detailed explanation for engineers:**
        This is a lightweight substitute for a full transformer encoder.  It
        works by tokenising the text on whitespace, computing a character-level
        hash for each token, and using those hashes to index into a random
        projection matrix generated deterministically from ``seed``.

        The resulting vector captures some vocabulary statistics but none of the
        semantic structure that a proper encoder would produce.  However, for
        the purpose of testing JEPA v24's domain-reweighting loss, it is
        sufficient: correct and incorrect steps in the same domain share
        structural vocabulary patterns that the projection captures.

    Args:
        text: The step text to embed.
        dim: Output embedding dimension (default: EMBED_DIM=256).
        seed: PRNG seed for the projection matrix.

    Returns:
        Unit-normed numpy float32 array of shape (dim,).
    """
    rng = np.random.RandomState(seed)
    proj = rng.randn(10000, dim).astype(np.float32)

    tokens = text.lower().split()
    vec = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        idx = hash(token) % 10000
        vec += proj[idx]
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# JEPA v24 model: dual-head (correctness + domain classifier)
# ---------------------------------------------------------------------------

def _init_v24_params(key: jax.Array) -> dict[str, jax.Array]:
    """Initialise v24 dual-head MLP parameters with He (Kaiming) initialisation.

    **Architecture:**
        Shared trunk: EMBED_DIM → HIDDEN1 → HIDDEN2
        Correctness head: HIDDEN2 → 1 (binary)
        Domain head: HIDDEN2 → N_DOMAINS (4-class softmax)

    **Why two heads:**
        The DG-PRM paper (arXiv 2507.17849) shows that training a domain
        classifier jointly with the correctness predictor acts as a
        multi-task regulariser: the shared trunk learns domain-invariant
        features while the domain head provides gradient signal that
        prevents the trunk from collapsing to a single-domain representation.

    Returns:
        Dict of JAX arrays: w1, b1, w2, b2, w_corr, b_corr, w_dom, b_dom.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    w1 = jax.random.normal(k1, (EMBED_DIM, HIDDEN1)) * np.sqrt(2.0 / EMBED_DIM)
    w2 = jax.random.normal(k2, (HIDDEN1, HIDDEN2)) * np.sqrt(2.0 / HIDDEN1)
    w_corr = jax.random.normal(k3, (HIDDEN2, N_CORRECTNESS)) * np.sqrt(2.0 / HIDDEN2)
    w_dom = jax.random.normal(k4, (HIDDEN2, N_DOMAINS)) * np.sqrt(2.0 / HIDDEN2)
    return {
        "w1": w1.astype(jnp.float32),
        "b1": jnp.zeros((HIDDEN1,), dtype=jnp.float32),
        "w2": w2.astype(jnp.float32),
        "b2": jnp.zeros((HIDDEN2,), dtype=jnp.float32),
        "w_corr": w_corr.astype(jnp.float32),
        "b_corr": jnp.zeros((N_CORRECTNESS,), dtype=jnp.float32),
        "w_dom": w_dom.astype(jnp.float32),
        "b_dom": jnp.zeros((N_DOMAINS,), dtype=jnp.float32),
    }


def _forward_v24(
    params: dict[str, jax.Array], x: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Forward pass for the v24 dual-head model.

    **Detailed explanation:**
        1. Shared trunk: x → ReLU(xW1+b1) → ReLU(h1W2+b2) → h2
        2. Correctness head: sigmoid(h2 @ w_corr + b_corr) → P(correct) in [0,1]
        3. Domain head: softmax(h2 @ w_dom + b_dom) → P(domain) as 4-dim vector

    Args:
        params: Parameter dict from _init_v24_params.
        x: Input embedding, shape (EMBED_DIM,) or (batch, EMBED_DIM).

    Returns:
        Tuple (corr_prob, dom_prob) where corr_prob has shape (..., 1) and
        dom_prob has shape (..., N_DOMAINS).
    """
    h1 = jax.nn.relu(x @ params["w1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["w2"] + params["b2"])
    corr_logit = h2 @ params["w_corr"] + params["b_corr"]
    dom_logit = h2 @ params["w_dom"] + params["b_dom"]
    return jax.nn.sigmoid(corr_logit), jax.nn.softmax(dom_logit, axis=-1)


def _compute_loss(
    params: dict[str, jax.Array],
    x_batch: jax.Array,
    y_corr: jax.Array,
    y_dom: jax.Array,
    dream_weights: jax.Array,
    x_pos: jax.Array,
    x_neg: jax.Array,
    delta_weights: jax.Array,
) -> jax.Array:
    """Compute the combined DG-PRM + DreamPRM + ΔEnergy loss.

    **Loss components:**

    1. BCE correctness loss (DreamPRM-weighted):
       For each sample i, BCE(P(correct|xᵢ), yᵢ) × dream_weights[domain(xᵢ)]
       Domains with high historical loss get multiplied weight, forcing the
       model to focus on them.

    2. Cross-entropy domain classification loss:
       CE(P(domain|xᵢ), true_domain(xᵢ))
       Keeps the shared trunk domain-discriminative.

    3. Triplet loss (ΔEnergy-weighted):
       For each pair (positive xᵢ, negative xⱼ in same domain):
       max(0, energy(xᵢ) - energy(xⱼ) + margin) × delta_weight(xᵢ, xⱼ)
       where energy(x) = 1 - P(correct|x) and delta_weight is clamped to [0.5, 3.0].

    Args:
        params: Model parameters.
        x_batch: Batch embeddings, shape (batch, EMBED_DIM).
        y_corr: Binary correctness labels, shape (batch, 1).
        y_dom: Domain label indices, shape (batch,).
        dream_weights: Per-sample DreamPRM weights, shape (batch,).
        x_pos: Positive (correct) anchor embeddings, shape (triplet_n, EMBED_DIM).
        x_neg: Negative (incorrect) anchor embeddings, shape (triplet_n, EMBED_DIM).
        delta_weights: ΔEnergy triplet weights, shape (triplet_n,).

    Returns:
        Scalar total loss.
    """
    corr_prob, dom_prob = _forward_v24(params, x_batch)

    # 1. DreamPRM-weighted BCE correctness loss
    corr_loss_per = optax.sigmoid_binary_cross_entropy(
        corr_prob.squeeze(-1), y_corr.squeeze(-1)
    )  # (batch,)
    corr_loss = jnp.mean(corr_loss_per * dream_weights)

    # 2. Domain classification cross-entropy loss
    # Clip probabilities for numerical stability before taking log
    dom_log_prob = jnp.log(jnp.clip(dom_prob, 1e-7, 1.0))
    dom_loss = -jnp.mean(dom_log_prob[jnp.arange(len(y_dom)), y_dom])

    # 3. ΔEnergy-weighted triplet loss
    # energy(x) = 1 - P(correct|x) — higher energy means more likely incorrect
    corr_pos, _ = _forward_v24(params, x_pos)
    corr_neg, _ = _forward_v24(params, x_neg)
    energy_pos = 1.0 - corr_pos.squeeze(-1)  # (triplet_n,)
    energy_neg = 1.0 - corr_neg.squeeze(-1)  # (triplet_n,)
    triplet_raw = jnp.maximum(0.0, energy_pos - energy_neg + TRIPLET_MARGIN)
    triplet_loss = jnp.mean(triplet_raw * delta_weights)

    return corr_loss + 0.3 * dom_loss + 0.5 * triplet_loss


_grad_loss_v24 = jax.jit(jax.value_and_grad(_compute_loss))


def _build_triplets(
    X: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    params: dict[str, jax.Array],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (positive, negative) anchor pairs for ΔEnergy triplet loss.

    **Strategy:**
        For each domain, pair each correct step (label=1) with each incorrect
        step (label=0) in the same domain.  Compute the energy delta for each
        pair and clamp to [DELTA_ENERGY_MIN, DELTA_ENERGY_MAX].

        Pairs within the same domain are important because cross-domain pairs
        would confound the domain-reweighting signal: a GSM8K incorrect step
        should NOT be used as a negative example for ARC.

    Returns:
        Tuple of (x_pos, x_neg, delta_weights), each shape (n_triplets, EMBED_DIM)
        or (n_triplets,) for delta_weights.
    """
    pos_list = []
    neg_list = []
    delta_list = []

    for d_idx in range(N_DOMAINS):
        pos_mask = (labels == 1) & (domains == d_idx)
        neg_mask = (labels == 0) & (domains == d_idx)
        X_pos = X[pos_mask]
        X_neg = X[neg_mask]
        if len(X_pos) == 0 or len(X_neg) == 0:
            continue
        # Pair each positive with its matched negative (zip, truncate to shorter list).
        for i in range(min(len(X_pos), len(X_neg))):
            xp = X_pos[i]
            xn = X_neg[i]
            # Energy = 1 - P(correct). Compute current model energy.
            cp, _ = _forward_v24(params, jnp.asarray(xp))
            cn, _ = _forward_v24(params, jnp.asarray(xn))
            ep = float(1.0 - cp.squeeze())
            en = float(1.0 - cn.squeeze())
            delta = abs(en - ep)
            # Clamp delta weight
            dw = float(np.clip(delta, DELTA_ENERGY_MIN, DELTA_ENERGY_MAX))
            pos_list.append(xp)
            neg_list.append(xn)
            delta_list.append(dw)

    if not pos_list:
        dummy = np.zeros((1, EMBED_DIM), dtype=np.float32)
        return dummy, dummy, np.ones(1, dtype=np.float32)

    return (
        np.array(pos_list, dtype=np.float32),
        np.array(neg_list, dtype=np.float32),
        np.array(delta_list, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_jepa_v24(
    pairs: list[dict[str, Any]],
    n_epochs: int = N_EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    val_fraction: float = VAL_FRACTION,
    seed: int = 42,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Train the JEPA v24 dual-head model with DG-PRM + DreamPRM + ΔEnergy.

    **Detailed explanation for engineers:**
        1. Embed all pair texts into EMBED_DIM-dimensional vectors.
        2. Stratified 80/20 train/val split (stratified by domain × label).
        3. Train for n_epochs epochs:
           a. Build triplets from current model energy.
           b. Compute combined loss: BCE + domain CE + ΔEnergy triplet.
           c. Apply Adam gradient step.
        4. After training, evaluate per-domain AUC on held-out val set.

    Args:
        pairs: Output of build_balanced_corpus() — list of {text, label, domain, domain_idx}.
        n_epochs: Number of training epochs.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        val_fraction: Fraction of data to hold out for validation.
        seed: PRNG seed.

    Returns:
        Tuple (final_params, training_log).
    """
    rng = np.random.RandomState(seed)

    # 1. Embed all texts
    X = np.array([_embed_text(p["text"]) for p in pairs], dtype=np.float32)
    Y_corr = np.array([[p["label"]] for p in pairs], dtype=np.float32)
    Y_dom = np.array([p["domain_idx"] for p in pairs], dtype=np.int32)
    Y_label_flat = np.array([p["label"] for p in pairs], dtype=np.int32)

    # 2. Stratified train/val split by domain × label combination
    strat_key = Y_dom * 2 + Y_label_flat
    unique_strats = np.unique(strat_key)
    train_idx_list, val_idx_list = [], []
    for s in unique_strats:
        idx = np.where(strat_key == s)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_idx_list.extend(idx[:n_val].tolist())
        train_idx_list.extend(idx[n_val:].tolist())

    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)

    X_tr, Y_corr_tr, Y_dom_tr = X[train_idx], Y_corr[train_idx], Y_dom[train_idx]
    X_val, Y_corr_val, Y_dom_val = X[val_idx], Y_corr[val_idx], Y_dom[val_idx]

    # 3. Build DreamPRM per-sample weights (proportional to domain loss weight)
    dream_weights_tr = np.array(
        [DREAM_PRM_WEIGHTS[DOMAIN_NAMES[d]] for d in Y_dom_tr], dtype=np.float32
    )

    # 4. Initialise model and Adam optimizer
    key = jax.random.PRNGKey(seed)
    params = _init_v24_params(key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def _step(p, state, xb, yc, yd, dw, xpos, xneg, delt):
        loss, grads = _grad_loss_v24(p, xb, yc, yd, dw, xpos, xneg, delt)
        updates, new_state = optimizer.update(grads, state, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_state, loss

    train_losses = []
    val_losses = []
    n_tr = len(X_tr)

    for epoch in range(n_epochs):
        perm = rng.permutation(n_tr)
        X_sh = X_tr[perm]
        Yc_sh = Y_corr_tr[perm]
        Yd_sh = Y_dom_tr[perm]
        Dw_sh = dream_weights_tr[perm]

        # Rebuild triplets every 50 epochs to track current model energy
        if epoch % 50 == 0:
            label_tr = (Y_corr_tr.squeeze() > 0.5).astype(np.int32)
            x_pos, x_neg, delta_weights = _build_triplets(
                X_tr, label_tr, Y_dom_tr, params
            )

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_tr, batch_size):
            end = min(start + batch_size, n_tr)
            xb = jnp.asarray(X_sh[start:end])
            yc = jnp.asarray(Yc_sh[start:end])
            yd = jnp.asarray(Yd_sh[start:end])
            dw = jnp.asarray(Dw_sh[start:end])
            # Select triplet batch (cycle through available triplets)
            n_trip = len(x_pos)
            t_start = (start // batch_size * batch_size) % n_trip
            t_end = min(t_start + batch_size, n_trip)
            xpos_b = jnp.asarray(x_pos[t_start:t_end])
            xneg_b = jnp.asarray(x_neg[t_start:t_end])
            delt_b = jnp.asarray(delta_weights[t_start:t_end])
            params, opt_state, loss = _step(
                params, opt_state, xb, yc, yd, dw, xpos_b, xneg_b, delt_b
            )
            epoch_loss += float(loss)
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # Compute validation loss (same triplet batch reused for speed)
        val_loss_val = float(
            _compute_loss(
                params,
                jnp.asarray(X_val),
                jnp.asarray(Y_corr_val),
                jnp.asarray(Y_dom_val),
                jnp.ones(len(X_val), dtype=jnp.float32),
                jnp.asarray(x_pos[:min(len(x_pos), len(X_val))]),
                jnp.asarray(x_neg[:min(len(x_neg), len(X_val))]),
                jnp.asarray(delta_weights[:min(len(delta_weights), len(X_val))]),
            )
        )
        val_losses.append(val_loss_val)

    # 5. Evaluate per-domain AUC on validation set
    corr_probs_val, dom_probs_val = _forward_v24(params, jnp.asarray(X_val))
    corr_probs_np = np.array(corr_probs_val.squeeze(-1))
    dom_probs_np = np.array(dom_probs_val)
    Y_corr_val_flat = Y_corr_val.squeeze(-1)

    # Apply DG-PRM domain weight at inference: multiply correctness score by
    # the weight of the predicted domain (argmax of domain classifier output).
    predicted_domains = np.argmax(dom_probs_np, axis=-1)
    domain_weight_vec = np.array(
        [DG_PRM_DOMAIN_WEIGHTS[DOMAIN_NAMES[d]] for d in predicted_domains],
        dtype=np.float32,
    )
    adjusted_probs = np.clip(corr_probs_np * domain_weight_vec, 0.0, 1.0)

    auc_per_domain: dict[str, float] = {}
    for d_idx, domain in enumerate(DOMAIN_NAMES):
        mask = Y_dom_val == d_idx
        if mask.sum() == 0:
            auc_per_domain[domain] = 0.5
            continue
        y_true = Y_corr_val_flat[mask]
        y_pred = adjusted_probs[mask]
        if len(np.unique(y_true)) < 2:
            auc_per_domain[domain] = 0.5
            continue
        auc_per_domain[domain] = float(roc_auc_score(y_true, y_pred))

    log = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "auc_per_domain": auc_per_domain,
        "n_train": int(n_tr),
        "n_val": int(len(X_val)),
    }
    return params, log


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def compute_honest_verdict(
    auc_gsm8k: float,
    auc_humaneval: float,
    auc_arc: float,
    auc_svamp: float,
    overall_ood_auc: float,
    min_domain_auc: float,
) -> str:
    """Map per-domain AUC results to an honest verdict string.

    **Logic (in priority order):**
        1. jepa_v24_domain_balanced: all per-domain AUCs > 0.55 (the target gate)
        2. jepa_v24_improvement: overall OOD AUC > 0.65 but one domain still below gate
        3. jepa_v24_arc_improved: ARC AUC > 0.40 (progress, even if below gate)
        4. jepa_v24_still_unbalanced: ARC AUC <= 0.40 (failed to fix the v23 root cause)

    Args:
        auc_gsm8k: ROC-AUC for GSM8K domain on val set.
        auc_humaneval: ROC-AUC for HumanEval domain on val set.
        auc_arc: ROC-AUC for ARC-Challenge domain on val set.
        auc_svamp: ROC-AUC for SVAMP domain on val set.
        overall_ood_auc: Mean AUC across all four domains.
        min_domain_auc: Minimum AUC across all four domains.

    Returns:
        One of: "jepa_v24_domain_balanced", "jepa_v24_improvement",
                "jepa_v24_arc_improved", "jepa_v24_still_unbalanced".
    """
    if min_domain_auc > 0.55:
        return "jepa_v24_domain_balanced"
    if overall_ood_auc > 0.65:
        return "jepa_v24_improvement"
    if auc_arc > 0.40:
        return "jepa_v24_arc_improved"
    return "jepa_v24_still_unbalanced"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 834: JEPA v24 DG-PRM domain-balanced training and evaluation."""
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # --- Build corpus ---
    with tmpl.phase("corpus_build"):
        corpus = build_balanced_corpus()
        domain_counts = {d: 0 for d in DOMAIN_NAMES}
        for p in corpus:
            domain_counts[p["domain"]] += 1

    corpus_composition = {
        domain: {"total": count, "correct": count // 2, "incorrect": count // 2}
        for domain, count in domain_counts.items()
    }

    # --- Train ---
    with tmpl.phase("training", n_epochs=N_EPOCHS, n_pairs=len(corpus)):
        params, train_log = train_jepa_v24(corpus, n_epochs=N_EPOCHS)
        tmpl.checkpoint_save({"model_trained": True}, step=N_EPOCHS)

    # --- Extract metrics ---
    auc_gsm8k = train_log["auc_per_domain"].get("gsm8k", 0.5)
    auc_humaneval = train_log["auc_per_domain"].get("humaneval", 0.5)
    auc_arc = train_log["auc_per_domain"].get("arc", 0.5)
    auc_svamp = train_log["auc_per_domain"].get("svamp", 0.5)
    overall_ood_auc = float(np.mean([auc_gsm8k, auc_humaneval, auc_arc, auc_svamp]))
    min_domain_auc = float(min(auc_gsm8k, auc_humaneval, auc_arc, auc_svamp))

    verdict = compute_honest_verdict(
        auc_gsm8k, auc_humaneval, auc_arc, auc_svamp, overall_ood_auc, min_domain_auc
    )

    # --- Write result ---
    artifact = tmpl.build_result(
        {
            "auc_gsm8k": auc_gsm8k,
            "auc_humaneval": auc_humaneval,
            "auc_arc": auc_arc,
            "auc_svamp": auc_svamp,
            "overall_ood_auc": overall_ood_auc,
            "min_domain_auc": min_domain_auc,
            "retro_jepa_ood_improving": bool(min_domain_auc > 0.55),
            "honest_verdict": verdict,
            "domain_weights_used": DG_PRM_DOMAIN_WEIGHTS,
            "dream_prm_weights_used": DREAM_PRM_WEIGHTS,
            "corpus_composition": corpus_composition,
            "n_training_pairs": len(corpus),
            "n_train": train_log["n_train"],
            "n_val": train_log["n_val"],
            "n_epochs": N_EPOCHS,
            "final_train_loss": train_log["train_losses"][-1] if train_log["train_losses"] else None,
            "final_val_loss": train_log["val_losses"][-1] if train_log["val_losses"] else None,
            "arc_auc_v23_baseline": 0.04,
        },
        status="success",
        decision_class="verify",
    )

    out_path = Path(_REPO_ROOT) / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
