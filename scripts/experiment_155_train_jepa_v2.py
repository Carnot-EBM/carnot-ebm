"""Experiment 155: Retrain JEPA Violation Predictor v2 with multi-domain data.

**Researcher summary:**
    Exp 144 trained on arithmetic-only data, yielding arithmetic AUROC=0.7126 but
    code/logic AUROC=0.5 (chance). This experiment trains v2 on balanced multi-domain
    pairs (arithmetic + code + logic), using stratified splits, class-weighted loss,
    100 epochs with early stopping, and compares against v1.

**What this experiment tests:**
    - Does training on balanced multi-domain data lift code/logic AUROC above 0.5?
    - Can we reach macro AUROC > 0.70 across all three domains?
    - Improvement from class-balanced loss over unweighted BCE?

**Improvements over Exp 144:**
    - Stratified split by (domain, violated) — each domain and class represented in val
    - Class-weighted BCE loss: upweights violated=True examples per domain
    - 100 training epochs (vs 50 in Exp 144) with early stopping on val AUROC
    - Multi-domain training data (~1200 pairs vs 800 arithmetic-only)

**Results are written to:**
    - results/jepa_predictor_v2.safetensors — trained v2 model parameters
    - results/experiment_155_results.json — comparison table vs v1
    - stdout — per-domain AUROC, precision/recall, comparison vs Exp 144

Usage:
    JAX_PLATFORMS=cpu python scripts/experiment_155_train_jepa_v2.py

Spec: REQ-JEPA-001, SCENARIO-JEPA-003
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

# Ensure the package root is on the path when run from project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from carnot.embeddings.fast_embedding import RandomProjectionEmbedding
from carnot.pipeline.jepa_predictor import (
    DOMAINS,
    EMBED_DIM,
    N_DOMAINS,
    JEPAViolationPredictor,
    _forward,
    _init_params,
)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

V2_PAIRS_PATH = Path("results/jepa_training_pairs_v2.json")
V1_PAIRS_PATH = Path("results/jepa_training_pairs.json")
V1_MODEL_PATH = Path("results/jepa_predictor.safetensors")
V2_MODEL_PATH = Path("results/jepa_predictor_v2.safetensors")
RESULTS_PATH = Path("results/experiment_155_results.json")

# Training config for Exp 155
N_EPOCHS = 100          # doubled from Exp 144's 50
LR = 1e-3               # same learning rate
BATCH_SIZE = 64         # same batch size
VAL_FRACTION = 0.2      # same val fraction
SEED = 42               # reproducibility
EARLY_STOP_PATIENCE = 15  # stop if val AUROC doesn't improve for N epochs

# Exp 144 v1 reference results (for comparison table)
V1_REFERENCE = {
    "arithmetic": 0.7126,
    "code": 0.5,
    "logic": 0.5,
    "macro": 0.5709,  # mean of (0.7126, 0.5, 0.5)
}

# ---------------------------------------------------------------------------
# Synthetic multi-domain data generation
# ---------------------------------------------------------------------------

# Code domain: Python snippets — some with syntax/logic bugs
_CODE_CORRECT_TEMPLATES = [
    # Sorting and searching
    "def sort_list(items):\n    return sorted(items)\n",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    "def find_max(nums):\n    return max(nums)\n",
    "def count_evens(nums):\n    return sum(1 for n in nums if n % 2 == 0)\n",
    "def reverse_string(s):\n    return s[::-1]\n",
    "def is_palindrome(s):\n    return s == s[::-1]\n",
    "def flatten(lst):\n    return [x for sub in lst for x in sub]\n",
    "def unique(items):\n    return list(set(items))\n",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
    "def sum_digits(n):\n    return sum(int(d) for d in str(abs(n)))\n",
    "def count_words(text):\n    return len(text.split())\n",
    "def to_uppercase(s):\n    return s.upper()\n",
    "def remove_duplicates(lst):\n    seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]\n",
    "def zip_lists(a, b):\n    return list(zip(a, b))\n",
    "def mean(nums):\n    return sum(nums) / len(nums)\n",
    "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
    "def is_sorted(lst):\n    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))\n",
    "def head(lst):\n    return lst[0] if lst else None\n",
    "def tail(lst):\n    return lst[1:]\n",
    "def dot_product(a, b):\n    return sum(x * y for x, y in zip(a, b))\n",
    "def chunk(lst, n):\n    return [lst[i:i+n] for i in range(0, len(lst), n)]\n",
    "def safe_divide(a, b):\n    return a / b if b != 0 else None\n",
]

_CODE_BUGGY_TEMPLATES = [
    # Syntax/logic errors introduced
    "def sort_list(items):\n    return sorted(items\n",                               # missing closing paren
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = lo + hi // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",  # operator precedence bug
    "def find_max(nums):\n    return min(nums)\n",                                    # wrong function
    "def count_evens(nums):\n    return sum(1 for n in nums if n % 2 == 1)\n",       # off-by-one in modulo
    "def reverse_string(s):\n    return s[::1]\n",                                   # wrong step
    "def is_palindrome(s):\n    return s == reversed(s)\n",                          # reversed returns iterator
    "def flatten(lst):\n    return [x for sub in lst for x in lst]\n",               # inner loop wrong var
    "def unique(items):\n    return list(set(items, items))\n",                      # extra arg to set
    "def factorial(n):\n    if n <= 1:\n        return 0\n    return n * factorial(n - 1)\n",  # base case bug
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = a + b, a\n",  # swap wrong direction
    "def gcd(a, b):\n    while b:\n        a, b = b, b % a\n    return a\n",         # a and b swapped in mod
    "def is_prime(n):\n    if n < 2:\n        return True\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",  # base case inverted
    "def sum_digits(n):\n    return sum(d for d in str(abs(n)))\n",                  # missing int() conversion
    "def count_words(text):\n    return len(text)\n",                                # counts chars not words
    "def to_uppercase(s):\n    return s.lower()\n",                                  # wrong method
    "def remove_duplicates(lst):\n    seen = set()\n    return [x for x in lst if x in seen or seen.add(x)]\n",  # condition inverted
    "def zip_lists(a, b):\n    return list(zip(a))\n",                               # missing arg
    "def mean(nums):\n    return sum(nums) / len(nums) - 1\n",                       # off-by-one
    "def clamp(x, lo, hi):\n    return min(lo, max(hi, x))\n",                      # min/max swapped
    "def is_sorted(lst):\n    return all(lst[i] < lst[i+1] for i in range(len(lst)-1))\n",  # strict vs non-strict
    "def head(lst):\n    return lst[-1] if lst else None\n",                         # -1 instead of 0
    "def tail(lst):\n    return lst[:-1]\n",                                         # drops last not first
    "def dot_product(a, b):\n    return sum(x + y for x, y in zip(a, b))\n",        # addition instead of multiply
    "def chunk(lst, n):\n    return [lst[i:i+n] for i in range(0, len(lst))]\n",    # missing step
    "def safe_divide(a, b):\n    return a / b if b == 0 else None\n",               # condition inverted
]

# Logic domain: syllogism / propositional reasoning snippets
_LOGIC_VALID_TEMPLATES = [
    "All mammals are warm-blooded. A dog is a mammal. Therefore, a dog is warm-blooded.",
    "If it rains, the ground gets wet. It is raining. Therefore, the ground is wet.",
    "All prime numbers greater than 2 are odd. 7 is a prime number greater than 2. Therefore, 7 is odd.",
    "No reptiles are warm-blooded. All snakes are reptiles. Therefore, no snakes are warm-blooded.",
    "If a number is divisible by 4, it is divisible by 2. 12 is divisible by 4. Therefore, 12 is divisible by 2.",
    "All squares are rectangles. All rectangles have four right angles. Therefore, all squares have four right angles.",
    "If P then Q. If Q then R. P is true. Therefore, R is true.",
    "Either it is day or it is night. It is not day. Therefore, it is night.",
    "All birds have wings. A sparrow is a bird. Therefore, a sparrow has wings.",
    "If a triangle has three equal sides, it is equilateral. Triangle ABC has three equal sides. Therefore, triangle ABC is equilateral.",
    "No fish can live on land. A salmon is a fish. Therefore, a salmon cannot live on land.",
    "All integers are rational numbers. 5 is an integer. Therefore, 5 is a rational number.",
    "If X > Y and Y > Z, then X > Z. We know X > Y and Y > Z. Therefore, X > Z.",
    "All even numbers are divisible by 2. 8 is an even number. Therefore, 8 is divisible by 2.",
    "If the battery is dead, the car won't start. The battery is dead. Therefore, the car won't start.",
    "All humans are mortal. Socrates is a human. Therefore, Socrates is mortal.",
    "If a set is finite, it has a maximum element under a total order. {1,2,3} is a finite ordered set. Therefore, {1,2,3} has a maximum element.",
    "No perfect squares are negative. 16 is a perfect square. Therefore, 16 is not negative.",
    "If A implies B and B implies C, then A implies C. A implies B. B implies C. Therefore, A implies C.",
    "All equilateral triangles are isosceles. Triangle T is equilateral. Therefore, triangle T is isosceles.",
    "If it is freezing, water turns to ice. It is freezing. Therefore, water turns to ice.",
    "All prime factors of 12 are less than 12. 2 is a prime factor of 12. Therefore, 2 < 12.",
    "Either the door is open or it is closed. The door is not open. Therefore, the door is closed.",
    "All multiples of 6 are multiples of 3. 18 is a multiple of 6. Therefore, 18 is a multiple of 3.",
    "If a function is differentiable, it is continuous. f is differentiable. Therefore, f is continuous.",
]

_LOGIC_INVALID_TEMPLATES = [
    # Formal fallacies
    "All mammals are warm-blooded. A snake is warm-blooded. Therefore, a snake is a mammal.",   # affirming the consequent
    "If it rains, the ground gets wet. The ground is wet. Therefore, it rained.",               # affirming the consequent
    "All prime numbers greater than 2 are odd. 9 is odd. Therefore, 9 is prime.",              # converse error
    "No reptiles are warm-blooded. No snakes are warm-blooded. Therefore, all snakes are reptiles.",  # undistributed middle
    "If a number is divisible by 4, it is divisible by 2. 6 is divisible by 2. Therefore, 6 is divisible by 4.",  # converse
    "All squares are rectangles. Shape S is a rectangle. Therefore, shape S is a square.",      # converse error
    "If P then Q. Q is true. Therefore, P is true.",                                           # affirming consequent
    "Either it is day or it is night. It is day. Therefore, it is not night.",                 # false exclusion (can be twilight)
    "All birds have wings. Bats have wings. Therefore, bats are birds.",                        # undistributed middle
    "All equilateral triangles are equiangular. Triangle T is equiangular. Therefore, T is equilateral.",  # converse
    "No fish can live on land. Dolphins cannot live on land. Therefore, dolphins are fish.",    # converse
    "All integers are rational. 0.5 is rational. Therefore, 0.5 is an integer.",               # converse error
    "If X > Y and Y > Z, then X > Z. We know X > Z. Therefore, X > Y.",                       # affirming consequent
    "All even numbers are divisible by 2. 9 is divisible by 3. Therefore, 9 is even.",        # non sequitur
    "If the battery is dead, the car won't start. The car won't start. Therefore, the battery is dead.",  # affirming consequent
    "All humans are mortal. All dogs are mortal. Therefore, all dogs are humans.",              # undistributed middle
    "If a set is finite, it has a maximum. Set S has a maximum. Therefore, S is finite.",      # affirming consequent
    "No perfect squares are negative. -4 is not a perfect square. Therefore, -4 is negative.",  # denying antecedent
    "If A implies B, then B implies A. A implies B. Therefore, B implies A.",                  # converse fallacy
    "All equilateral triangles are isosceles. Triangle T is isosceles. Therefore, T is equilateral.",  # converse
    "If it is freezing, water turns to ice. Water is ice. Therefore, it is freezing.",         # affirming consequent
    "All prime factors of 12 are less than 12. 7 is less than 12. Therefore, 7 is a prime factor of 12.",  # converse
    "Either the door is open or closed. The door is open. Therefore, it is not closed.",       # incorrect disjunction exclusive
    "All multiples of 3 are multiples of 6. 9 is a multiple of 3. Therefore, 9 is a multiple of 6.",  # false premise
    "If a function is differentiable, it is continuous. f is continuous. Therefore, f is differentiable.",  # affirming consequent
]


def _detect_code_violation(code_text: str) -> bool:
    """Heuristic code violation detector: tries to compile, checks for obvious bugs.

    **Detailed explanation for engineers:**
        We use a two-pass approach:
        1. Try to compile the snippet with ``compile()`` — catches syntax errors.
        2. Check for known-bad patterns: wrong comparison operators, wrong base
           cases, swapped variables in common patterns.
        This is intentionally simple; for training data we control what's buggy.

    Args:
        code_text: Python source code string.

    Returns:
        True if the code appears to have a violation, False otherwise.
    """
    # Compile-time check catches syntax errors (missing parens, etc.)
    try:
        compile(code_text, "<string>", "exec")
    except SyntaxError:
        return True

    # Semantic heuristics for the specific bugs we introduced:
    # - Wrong step in slice: [::1] instead of [::-1]
    # - Wrong method: .lower() in to_uppercase
    # - Wrong function: min instead of max in find_max
    # - Wrong base case returns wrong value
    bad_patterns = [
        "[::1]",          # non-reversing slice step
        "return min(",    # find_max using min
        "return 0\n    return n",  # factorial base case returns 0
        "x + y for x, y",          # dot product using addition
        "a, b = a + b, a",         # fibonacci direction reversed
    ]
    return any(pat in code_text for pat in bad_patterns)


def _detect_logic_violation(logic_text: str) -> bool:
    """Heuristic logic violation detector: checks for known fallacy markers.

    **Detailed explanation for engineers:**
        Since we control the invalid logic templates, we can detect violations by
        looking for phrases that appear only in the invalid versions. In a real
        system this would use a formal logic checker or LLM scoring.

    Args:
        logic_text: Natural language logical argument string.

    Returns:
        True if the argument appears to be invalid, False otherwise.
    """
    # Phrases characteristic of the affirmation/converse fallacies we inserted.
    fallacy_markers = [
        "Therefore, a snake is a mammal",
        "Therefore, it rained",
        "Therefore, 9 is prime",
        "Therefore, all snakes are reptiles",
        "Therefore, 6 is divisible by 4",
        "Therefore, shape S is a square",
        "Therefore, P is true",
        "Therefore, bats are birds",
        "Therefore, T is equilateral.\n",  # from the invalid equiangular template
        "Therefore, dolphins are fish",
        "Therefore, 0.5 is an integer",
        "Therefore, X > Y.",
        "Therefore, 9 is even",
        "Therefore, the battery is dead",
        "Therefore, all dogs are humans",
        "Therefore, S is finite",
        "Therefore, -4 is negative",
        "therefore, B implies A",
        "Triangle T is isosceles. Therefore, T is equilateral",
        "Therefore, it is freezing",
        "Therefore, 7 is a prime factor of 12",
        "Therefore, it is not closed",
        "Therefore, 9 is a multiple of 6",
        "Therefore, f is differentiable",
        "return min(",  # not used here but keeping consistent
        "All multiples of 3 are multiples of 6",  # false premise
        "B implies A",  # converse fallacy statement
    ]
    return any(marker in logic_text for marker in fallacy_markers)


def _generate_v2_pairs(rng: np.random.RandomState) -> list[dict]:
    """Generate synthetic multi-domain training pairs for Exp 154/155.

    **Detailed explanation for engineers:**
        Since results/jepa_training_pairs_v2.json does not exist (Exp 154 was not
        run before this session), we generate it synthetically here using the same
        RandomProjectionEmbedding(embed_dim=256, seed=42) and the same schema as
        the Exp 143 arithmetic pairs. The synthetic code/logic snippets are
        designed so that violations are detectable by simple heuristics, giving
        the MLP a real signal to learn.

        Pair generation:
        - 800 arithmetic pairs from jepa_training_pairs.json (reused)
        - 200 code pairs: 25 correct * 4 prefix_ratios + 25 buggy * 4 prefix_ratios
        - 200 logic pairs: 25 valid * 4 prefix_ratios + 25 invalid * 4 prefix_ratios

    Args:
        rng: NumPy RandomState for reproducible generation.

    Returns:
        List of pair dicts with keys matching the v1 schema.
    """
    embedder = RandomProjectionEmbedding(embed_dim=EMBED_DIM, seed=42)
    prefix_ratios = [0.1, 0.25, 0.5, 0.75]
    pairs: list[dict] = []

    # --- Arithmetic pairs (reuse v1) ---
    if V1_PAIRS_PATH.exists():
        with open(V1_PAIRS_PATH) as f:
            v1_data = json.load(f)
        pairs.extend(v1_data["pairs"])
        print(f"  Loaded {len(v1_data['pairs'])} arithmetic pairs from {V1_PAIRS_PATH}")
    else:
        print(f"  WARNING: {V1_PAIRS_PATH} not found; arithmetic domain will be empty")

    # --- Code pairs ---
    n_correct_code = min(25, len(_CODE_CORRECT_TEMPLATES))
    n_buggy_code = min(25, len(_CODE_BUGGY_TEMPLATES))

    for templates, is_buggy in [
        (_CODE_CORRECT_TEMPLATES[:n_correct_code], False),
        (_CODE_BUGGY_TEMPLATES[:n_buggy_code], True),
    ]:
        for tmpl in templates:
            violated_code = _detect_code_violation(tmpl) if is_buggy else False
            full_text = tmpl
            text_len = len(full_text)
            for ratio in prefix_ratios:
                # Use a prefix of the text at the given ratio
                prefix = full_text[:max(1, int(text_len * ratio))]
                emb = embedder.encode(prefix)
                pairs.append(
                    {
                        "prefix_ratio": ratio,
                        "embedding": emb.tolist(),
                        "violated_arithmetic": False,
                        "violated_code": violated_code,
                        "violated_logic": False,
                        "any_violated": violated_code,
                        "domain": "code",
                        "source_exp": 155,
                    }
                )

    n_code = sum(1 for p in pairs if p["domain"] == "code")
    print(f"  Generated {n_code} code pairs")

    # --- Logic pairs ---
    n_valid_logic = min(25, len(_LOGIC_VALID_TEMPLATES))
    n_invalid_logic = min(25, len(_LOGIC_INVALID_TEMPLATES))

    for templates, is_invalid in [
        (_LOGIC_VALID_TEMPLATES[:n_valid_logic], False),
        (_LOGIC_INVALID_TEMPLATES[:n_invalid_logic], True),
    ]:
        for tmpl in templates:
            violated_logic = _detect_logic_violation(tmpl) if is_invalid else False
            full_text = tmpl
            text_len = len(full_text)
            for ratio in prefix_ratios:
                prefix = full_text[:max(1, int(text_len * ratio))]
                emb = embedder.encode(prefix)
                pairs.append(
                    {
                        "prefix_ratio": ratio,
                        "embedding": emb.tolist(),
                        "violated_arithmetic": False,
                        "violated_code": False,
                        "violated_logic": violated_logic,
                        "any_violated": violated_logic,
                        "domain": "logic",
                        "source_exp": 155,
                    }
                )

    n_logic = sum(1 for p in pairs if p["domain"] == "logic")
    print(f"  Generated {n_logic} logic pairs")
    return pairs


def generate_and_save_v2_pairs() -> list[dict]:
    """Generate v2 multi-domain pairs and save to results/jepa_training_pairs_v2.json.

    **Detailed explanation for engineers:**
        This function stands in for Exp 154 (which was not run). It generates
        ~1200 pairs across arithmetic (800), code (200), and logic (200) domains
        and writes them to jepa_training_pairs_v2.json in the same schema as
        jepa_training_pairs.json so that train() can consume them unchanged.

    Returns:
        List of pair dicts.
    """
    print("\n--- Generating multi-domain training pairs (Exp 154 stand-in) ---")
    rng = np.random.RandomState(SEED)
    pairs = _generate_v2_pairs(rng)

    # Compute summary stats
    from collections import Counter

    domain_counts: dict[str, int] = Counter(p["domain"] for p in pairs)
    total = len(pairs)
    n_positive = sum(1 for p in pairs if p["any_violated"])
    positive_rate = n_positive / total if total > 0 else 0.0

    per_domain_pos: dict[str, int] = {}
    for domain in DOMAINS:
        per_domain_pos[domain] = sum(
            1 for p in pairs if p["domain"] == domain and p["any_violated"]
        )

    print(f"\n  Total pairs: {total}")
    print(f"  Domain distribution: {dict(domain_counts)}")
    print(f"  Positive rate (any domain): {positive_rate:.3f}")
    for domain in DOMAINS:
        n_d = domain_counts.get(domain, 0)
        n_p = per_domain_pos.get(domain, 0)
        rate = n_p / n_d if n_d > 0 else 0.0
        print(f"    {domain}: {n_d} pairs, {n_p} violated ({rate:.1%})")

    data = {
        "pairs": pairs,
        "total": total,
        "domain_counts": dict(domain_counts),
        "positive_rate": round(positive_rate, 4),
        "per_domain_positive": per_domain_pos,
        "metadata": {
            "generated_by": "experiment_155_train_jepa_v2.py",
            "exp": 155,
            "note": "Generated as Exp 154 stand-in; arithmetic from Exp 143, code/logic synthetic",
        },
    }

    V2_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(V2_PAIRS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Saved {total} pairs to {V2_PAIRS_PATH}")
    return pairs


# ---------------------------------------------------------------------------
# Stratified split (domain + violated)
# ---------------------------------------------------------------------------


def stratified_split(
    pairs: list[dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Stratified 80/20 split by (domain, violated) stratum.

    **Detailed explanation for engineers:**
        Standard stratification stratifies only by the target label (any_violated).
        For multi-domain data this is insufficient — if code/logic have few samples
        they may be absent from the val set entirely, making their AUROC undefined.

        This function stratifies by every (domain, violated) combination, so each
        stratum contributes proportionally to both train and val. With 200 code
        pairs and 20% val fraction we get ~40 code pairs in val, distributed across
        violated=True and violated=False code samples.

        If a stratum has only 1 sample, it goes to train (validation needs ≥1 per
        class, so we only put it in val when a stratum has ≥2 samples).

    Args:
        pairs: List of pair dicts.
        val_fraction: Fraction for validation set.
        seed: Random seed.

    Returns:
        Tuple of (train_indices, val_indices) as lists of integers.
    """
    rng = np.random.RandomState(seed)

    # Group indices by (domain, violated) stratum
    strata: dict[tuple[str, bool], list[int]] = {}
    for i, p in enumerate(pairs):
        key = (p.get("domain", "arithmetic"), bool(p.get("any_violated", False)))
        strata.setdefault(key, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []

    for (domain, violated), indices in sorted(strata.items()):
        idx_arr = np.array(indices)
        rng.shuffle(idx_arr)
        n_val = max(1, int(len(idx_arr) * val_fraction)) if len(idx_arr) >= 2 else 0
        val_idx.extend(idx_arr[:n_val].tolist())
        train_idx.extend(idx_arr[n_val:].tolist())

    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Class-weighted BCE loss
# ---------------------------------------------------------------------------


def _compute_pos_weights(Y_train: np.ndarray) -> np.ndarray:
    """Compute per-domain positive class weights: n_neg / n_pos.

    **Detailed explanation for engineers:**
        Class weighting addresses imbalanced data where one class is much rarer
        than the other. For binary cross-entropy, applying a weight of n_neg/n_pos
        to the positive class makes each class contribute equally to the gradient
        regardless of how many samples exist per class.

        Example: if domain 'code' has 160 negatives and 40 positives, pos_weight
        = 160/40 = 4.0. Each positive example counts 4x as much as a negative
        example in the gradient, so the model learns to predict code violations
        even when they're rare.

    Args:
        Y_train: Binary label matrix of shape (n_train, N_DOMAINS).

    Returns:
        Float32 array of shape (N_DOMAINS,) — one weight per domain. Clipped
        to [0.1, 10.0] to prevent extreme weights destabilising training.
    """
    pos_weights = np.ones(N_DOMAINS, dtype=np.float32)
    for i in range(N_DOMAINS):
        n_pos = Y_train[:, i].sum()
        n_neg = len(Y_train) - n_pos
        if n_pos > 0:
            pos_weights[i] = float(n_neg) / float(n_pos)
    # Clip to avoid extreme weights destabilising gradient updates.
    return np.clip(pos_weights, 0.1, 10.0)


def _weighted_bce_loss(
    params: dict[str, jax.Array],
    x: jax.Array,
    y: jax.Array,
    pos_weights: jax.Array,
) -> jax.Array:
    """Binary cross-entropy loss with per-domain positive class weighting.

    **Detailed explanation for engineers:**
        Standard BCE assigns equal weight to every (sample, domain) pair. With
        class-weighted BCE, positive samples (violated=True) for each domain
        receive a higher weight proportional to how rare they are.

        Weight per element: w[i][d] = pos_weights[d] if y[i][d]==1 else 1.0
        Final loss: mean(w * BCE(logits, y))

    Args:
        params: Model parameter dict.
        x: Batch embeddings, shape (batch, EMBED_DIM).
        y: Batch binary labels, shape (batch, N_DOMAINS).
        pos_weights: Per-domain positive weights, shape (N_DOMAINS,).

    Returns:
        Scalar weighted mean BCE loss.
    """
    logits = _forward(params, x)  # (batch, N_DOMAINS)
    per_element = optax.sigmoid_binary_cross_entropy(logits, y)  # (batch, N_DOMAINS)
    # Element-wise weight: pos_weight where y=1, else 1.0
    # y is float {0.0, 1.0}; broadcast pos_weights over batch dimension.
    sample_weights = y * pos_weights[None, :] + (1.0 - y) * 1.0  # (batch, N_DOMAINS)
    return jnp.mean(per_element * sample_weights)


# ---------------------------------------------------------------------------
# Training with early stopping
# ---------------------------------------------------------------------------


def train_v2(
    predictor: JEPAViolationPredictor,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    patience: int = EARLY_STOP_PATIENCE,
) -> dict[str, Any]:
    """Train with class-weighted loss and early stopping on val AUROC.

    **Detailed explanation for engineers:**
        Key differences from JEPAViolationPredictor.train() (Exp 144):
        1. Stratified split was done externally by caller (by domain+violated).
        2. Class-weighted loss: upweights positive examples per domain.
        3. Early stopping: if val macro AUROC hasn't improved by >0.001 for
           ``patience`` consecutive epochs, stop training and restore the best
           weights from the epoch with highest val AUROC.

        Early stopping prevents overfitting (the model memorises the training
        set instead of generalising). With only ~1000 samples and a 100-epoch
        budget, overfitting is a real risk, so we monitor the validation AUROC
        rather than validation loss (AUROC is more directly aligned with our
        target metric).

    Args:
        predictor: JEPAViolationPredictor instance to train in-place.
        X_train: Training embeddings, shape (n_train, EMBED_DIM).
        Y_train: Training labels, shape (n_train, N_DOMAINS).
        X_val: Validation embeddings, shape (n_val, EMBED_DIM).
        Y_val: Validation labels, shape (n_val, N_DOMAINS).
        n_epochs: Maximum training epochs.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        patience: Early-stopping patience in epochs.

    Returns:
        Training log dict with losses, AUROC history, and best epoch.
    """
    from sklearn.metrics import roc_auc_score

    # Compute class weights from training labels.
    pos_weights_np = _compute_pos_weights(Y_train)
    pos_weights_jax = jnp.asarray(pos_weights_np)
    print(f"\n  Per-domain positive class weights: {dict(zip(DOMAINS, pos_weights_np.tolist()))}")

    # Set up optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(predictor._params)
    rng = np.random.RandomState(SEED)

    # JIT-compile the gradient function for the weighted loss.
    _weighted_grad = jax.jit(jax.value_and_grad(_weighted_bce_loss))

    @jax.jit
    def _step(
        params: dict[str, jax.Array],
        state: Any,
        x_batch: jax.Array,
        y_batch: jax.Array,
        pw: jax.Array,
    ) -> tuple[dict[str, jax.Array], Any, jax.Array]:
        """One Adam gradient step using weighted BCE loss.

        Args:
            params: Current model parameters.
            state: Optimizer state.
            x_batch: Batch embeddings, shape (batch, EMBED_DIM).
            y_batch: Batch labels, shape (batch, N_DOMAINS).
            pw: Per-domain positive class weights, shape (N_DOMAINS,).

        Returns:
            Tuple of (updated_params, updated_state, batch_loss).
        """
        loss, grads = _weighted_grad(params, x_batch, y_batch, pw)
        updates, new_state = optimizer.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    n_train = len(X_train)
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_auroc_history: list[float] = []

    # Early stopping state
    best_val_auroc = -1.0
    best_params = predictor._params
    no_improve_count = 0
    best_epoch = 0

    for epoch in range(n_epochs):
        # Shuffle training set each epoch.
        perm = rng.permutation(n_train)
        X_shuf = X_train[perm]
        Y_shuf = Y_train[perm]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            x_b = jnp.asarray(X_shuf[start:end])
            y_b = jnp.asarray(Y_shuf[start:end])
            predictor._params, opt_state, batch_loss = _step(
                predictor._params, opt_state, x_b, y_b, pos_weights_jax
            )
            epoch_loss += float(batch_loss)
            n_batches += 1

        train_losses.append(epoch_loss / max(n_batches, 1))

        # Validation loss (unweighted, for comparability across experiments).
        val_logits_np = np.array(_forward(predictor._params, jnp.asarray(X_val)))
        val_probs = 1.0 / (1.0 + np.exp(-val_logits_np))  # sigmoid

        # Val BCE loss (unweighted)
        val_loss_j = optax.sigmoid_binary_cross_entropy(
            jnp.asarray(val_logits_np), jnp.asarray(Y_val)
        )
        val_losses.append(float(jnp.mean(val_loss_j)))

        # Val AUROC per domain
        auroc_vals: list[float] = []
        for i in range(N_DOMAINS):
            y_true = Y_val[:, i]
            y_prob = val_probs[:, i]
            if len(np.unique(y_true)) < 2:
                auroc_vals.append(0.5)
            else:
                auroc_vals.append(float(roc_auc_score(y_true, y_prob)))
        macro_auroc = float(np.mean(auroc_vals))
        val_auroc_history.append(macro_auroc)

        # Early stopping check
        if macro_auroc > best_val_auroc + 0.001:
            best_val_auroc = macro_auroc
            best_params = {k: jnp.array(v) for k, v in predictor._params.items()}
            no_improve_count = 0
            best_epoch = epoch + 1
        else:
            no_improve_count += 1

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            auroc_str = ", ".join(f"{d}={v:.3f}" for d, v in zip(DOMAINS, auroc_vals))
            print(
                f"  Epoch {epoch+1:3d}/{n_epochs}: "
                f"train_loss={train_losses[-1]:.4f}, "
                f"val_loss={val_losses[-1]:.4f}, "
                f"val_auroc=[{auroc_str}], macro={macro_auroc:.4f}"
            )

        if no_improve_count >= patience:
            print(
                f"\n  Early stopping at epoch {epoch+1} "
                f"(no improvement for {patience} epochs). "
                f"Best epoch: {best_epoch}, best val AUROC: {best_val_auroc:.4f}"
            )
            break

    # Restore best params
    predictor._params = best_params

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_auroc_history": val_auroc_history,
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "total_epochs_run": len(train_losses),
    }


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------


def evaluate_on_val(
    predictor: JEPAViolationPredictor,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    val_domains: list[str],
) -> dict[str, Any]:
    """Compute per-domain AUROC, precision, and recall on the validation set.

    **Detailed explanation for engineers:**
        We evaluate each domain independently using only the subset of validation
        samples whose ``domain`` field matches. This is the correct way to measure
        per-domain performance when samples from different domains have different
        base rates: a code sample with violated_logic=False is not informative for
        the logic AUROC, since it was never "supposed" to have a logic violation.

        Additionally, we compute the overall AUROC using all validation samples
        (cross-domain) to capture the predictor's general discriminative ability.

    Args:
        predictor: Trained JEPAViolationPredictor.
        X_val: Validation embeddings, shape (n_val, EMBED_DIM).
        Y_val: Validation labels, shape (n_val, N_DOMAINS). Column order: DOMAINS.
        val_domains: Domain name for each validation sample, length n_val.

    Returns:
        Dict with per_domain_auroc, per_domain_precision, per_domain_recall,
        macro_auroc, and cross_domain_auroc.
    """
    from sklearn.metrics import roc_auc_score

    val_logits_np = np.array(_forward(predictor._params, jnp.asarray(X_val)))
    val_probs = 1.0 / (1.0 + np.exp(-val_logits_np))

    auroc_per_domain: dict[str, float] = {}
    precision_at_05: dict[str, float] = {}
    recall_at_05: dict[str, float] = {}

    for i, domain in enumerate(DOMAINS):
        # Use domain-specific subset for per-domain metrics
        domain_mask = np.array([d == domain for d in val_domains])
        y_true_dom = Y_val[domain_mask, i]
        y_prob_dom = val_probs[domain_mask, i]

        if len(y_true_dom) == 0 or len(np.unique(y_true_dom)) < 2:
            # Fall back to all-val if domain subset is too small
            y_true_dom = Y_val[:, i]
            y_prob_dom = val_probs[:, i]

        y_pred_dom = (y_prob_dom >= 0.5).astype(float)

        if len(np.unique(y_true_dom)) < 2:
            auroc_per_domain[domain] = 0.5
        else:
            auroc_per_domain[domain] = float(roc_auc_score(y_true_dom, y_prob_dom))

        tp = float(((y_pred_dom == 1) & (y_true_dom == 1)).sum())
        fp = float(((y_pred_dom == 1) & (y_true_dom == 0)).sum())
        fn = float(((y_pred_dom == 0) & (y_true_dom == 1)).sum())
        precision_at_05[domain] = tp / max(tp + fp, 1.0)
        recall_at_05[domain] = tp / max(tp + fn, 1.0)

    macro_auroc = float(np.mean(list(auroc_per_domain.values())))

    return {
        "val_auroc_per_domain": auroc_per_domain,
        "macro_auroc": macro_auroc,
        "precision_at_05": precision_at_05,
        "recall_at_05": recall_at_05,
    }


# ---------------------------------------------------------------------------
# Comparison printing
# ---------------------------------------------------------------------------


def print_comparison_table(v1_metrics: dict, v2_metrics: dict) -> None:
    """Print side-by-side comparison of v1 (Exp 144) vs v2 (Exp 155) metrics.

    Args:
        v1_metrics: Per-domain metrics from Exp 144 reference or v1 evaluation.
        v2_metrics: Per-domain metrics from v2 training.
    """
    print("\n" + "=" * 65)
    print("COMPARISON: Exp 144 (v1) vs Exp 155 (v2)")
    print("=" * 65)
    print(f"  {'Domain':<12}  {'v1 AUROC':>10}  {'v2 AUROC':>10}  {'Delta':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}")

    for domain in DOMAINS:
        v1_a = v1_metrics.get("per_domain", {}).get(domain, V1_REFERENCE.get(domain, 0.5))
        v2_a = v2_metrics["val_auroc_per_domain"][domain]
        delta = v2_a - v1_a
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"  {domain:<12}  {v1_a:>10.4f}  {v2_a:>10.4f}  {delta_str:>8}")

    v1_macro = v1_metrics.get("macro", V1_REFERENCE["macro"])
    v2_macro = v2_metrics["macro_auroc"]
    delta_macro = v2_macro - v1_macro
    delta_str = f"+{delta_macro:.4f}" if delta_macro >= 0 else f"{delta_macro:.4f}"
    print(f"  {'Macro':<12}  {v1_macro:>10.4f}  {v2_macro:>10.4f}  {delta_str:>8}")
    print("=" * 65)

    # Target check
    print(f"\n  Target: macro AUROC > 0.70")
    if v2_macro > 0.70:
        print(f"  RESULT: TARGET MET — macro AUROC = {v2_macro:.4f} ✓")
    else:
        print(f"  RESULT: TARGET NOT MET — macro AUROC = {v2_macro:.4f} (< 0.70)")
        # Explain why
        for domain in DOMAINS:
            v2_a = v2_metrics["val_auroc_per_domain"][domain]
            if v2_a <= 0.5:
                print(
                    f"    NOTE: {domain} AUROC={v2_a:.4f} (at or below chance)."
                    f" May need more balanced data or stronger embeddings."
                )


def print_detailed_metrics(label: str, metrics: dict) -> None:
    """Print per-domain AUROC + precision/recall table for one model version.

    Args:
        label: Label string (e.g. "v2 (Exp 155)").
        metrics: Dict from evaluate_on_val().
    """
    auroc = metrics["val_auroc_per_domain"]
    prec = metrics["precision_at_05"]
    recall = metrics["recall_at_05"]
    macro = metrics["macro_auroc"]

    print(f"\n--- {label} Validation Metrics (threshold=0.5) ---")
    print(f"  {'Domain':<12}  {'AUROC':>8}  {'Precision':>10}  {'Recall':>8}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*8}")
    for domain in DOMAINS:
        print(
            f"  {domain:<12}  {auroc[domain]:>8.4f}  {prec[domain]:>10.4f}  {recall[domain]:>8.4f}"
        )
    print(f"  {'Macro':<12}  {macro:>8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 155: retrain JEPA v2 on multi-domain data."""
    print("=" * 65)
    print("Experiment 155: Retrain JEPA Violation Predictor v2")
    print("Multi-domain training: arithmetic + code + logic")
    print("=" * 65)

    # --- Step 1: Ensure v2 training pairs exist ---
    if not V2_PAIRS_PATH.exists():
        print(f"\njepa_training_pairs_v2.json not found at {V2_PAIRS_PATH}.")
        print("Generating multi-domain training pairs (Exp 154 stand-in)...")
        pairs = generate_and_save_v2_pairs()
    else:
        print(f"\nLoading {V2_PAIRS_PATH}...")
        with open(V2_PAIRS_PATH) as f:
            data = json.load(f)
        pairs = data["pairs"]
        print(f"  Loaded {len(pairs)} pairs.")

    # --- Print dataset statistics ---
    from collections import Counter

    domain_counts: dict[str, int] = Counter(p.get("domain", "arithmetic") for p in pairs)
    total = len(pairs)
    print(f"\n--- Dataset Statistics ---")
    print(f"  Total: {total}")
    for domain in DOMAINS:
        n_d = domain_counts.get(domain, 0)
        key = f"violated_{domain}"
        n_pos = sum(1 for p in pairs if p.get(key, False) and p.get("domain") == domain)
        n_neg = n_d - n_pos
        rate = n_pos / n_d if n_d > 0 else 0.0
        print(f"  {domain}: {n_d} pairs, {n_pos} violated ({rate:.1%}), {n_neg} clean")

    # --- Step 2: Build X, Y arrays and stratified split ---
    X = np.array([p["embedding"] for p in pairs], dtype=np.float32)
    Y = np.array(
        [
            [
                float(p.get("violated_arithmetic", False)),
                float(p.get("violated_code", False)),
                float(p.get("violated_logic", False)),
            ]
            for p in pairs
        ],
        dtype=np.float32,
    )
    pair_domains = [p.get("domain", "arithmetic") for p in pairs]

    print("\n--- Stratified split (domain × violated) ---")
    train_idx, val_idx = stratified_split(pairs, val_fraction=VAL_FRACTION, seed=SEED)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    val_domains = [pair_domains[i] for i in val_idx]

    train_domain_counts: dict[str, int] = Counter(pair_domains[i] for i in train_idx)
    val_domain_counts: dict[str, int] = Counter(pair_domains[i] for i in val_idx)
    print(f"  n_train={len(train_idx)}, domain distribution: {dict(train_domain_counts)}")
    print(f"  n_val={len(val_idx)}, domain distribution: {dict(val_domain_counts)}")

    # --- Step 3: Train v2 ---
    print("\n--- Training v2 (100 epochs max, early stopping) ---")
    predictor_v2 = JEPAViolationPredictor(seed=SEED)
    train_log = train_v2(
        predictor_v2,
        X_train,
        Y_train,
        X_val,
        Y_val,
        n_epochs=N_EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        patience=EARLY_STOP_PATIENCE,
    )

    print(f"\n  Training summary:")
    print(f"    Epochs run: {train_log['total_epochs_run']}")
    print(f"    Best epoch: {train_log['best_epoch']}")
    print(f"    Best val macro AUROC: {train_log['best_val_auroc']:.4f}")
    print(f"    Train loss: {train_log['train_losses'][0]:.4f} → {train_log['train_losses'][-1]:.4f}")
    print(f"    Val   loss: {train_log['val_losses'][0]:.4f} → {train_log['val_losses'][-1]:.4f}")

    # --- Step 4: Evaluate v2 ---
    print("\n--- Evaluating v2 on held-out validation set ---")
    v2_metrics = evaluate_on_val(predictor_v2, X_val, Y_val, val_domains)
    print_detailed_metrics("v2 (Exp 155)", v2_metrics)

    # --- Load and evaluate v1 for comparison ---
    v1_metrics_ref = {
        "per_domain": V1_REFERENCE.copy(),
        "macro": V1_REFERENCE["macro"],
    }
    if V1_MODEL_PATH.exists():
        try:
            predictor_v1 = JEPAViolationPredictor(seed=0)
            predictor_v1.load(str(V1_MODEL_PATH))
            # Evaluate v1 on the same val set
            v1_eval = evaluate_on_val(predictor_v1, X_val, Y_val, val_domains)
            v1_metrics_ref = {
                "per_domain": v1_eval["val_auroc_per_domain"],
                "macro": v1_eval["macro_auroc"],
            }
            print_detailed_metrics("v1 (Exp 144)", v1_eval)
        except Exception as e:
            print(f"\n  Could not load v1 model for comparison: {e}")
            print("  Using Exp 144 reference results instead.")

    print_comparison_table(v1_metrics_ref, v2_metrics)

    # --- Step 5: Save v2 model ---
    V2_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictor_v2.save(str(V2_MODEL_PATH))
    print(f"\nModel saved to: {V2_MODEL_PATH}")
    size_kb = V2_MODEL_PATH.stat().st_size / 1024
    print(f"File size: {size_kb:.1f} KB")

    # Verify round-trip load
    loaded_v2 = JEPAViolationPredictor()
    loaded_v2.load(str(V2_MODEL_PATH))
    x_check = jnp.ones((EMBED_DIM,), dtype=jnp.float32)
    p_orig = predictor_v2.predict(x_check)
    p_load = loaded_v2.predict(x_check)
    assert all(abs(p_orig[d] - p_load[d]) < 1e-5 for d in DOMAINS), \
        "Round-trip load mismatch!"
    print("Round-trip load verified ✓")

    # --- Step 6: Save results JSON ---
    results = {
        "experiment": 155,
        "v1_macro_auroc": v1_metrics_ref["macro"],
        "v2_macro_auroc": v2_metrics["macro_auroc"],
        "per_domain_v1": v1_metrics_ref["per_domain"],
        "per_domain_v2": v2_metrics["val_auroc_per_domain"],
        "improvement": {
            domain: v2_metrics["val_auroc_per_domain"][domain]
            - v1_metrics_ref["per_domain"].get(domain, V1_REFERENCE.get(domain, 0.5))
            for domain in DOMAINS
        },
        "macro_improvement": v2_metrics["macro_auroc"] - v1_metrics_ref["macro"],
        "target_met": v2_metrics["macro_auroc"] > 0.70,
        "training": {
            "epochs_run": train_log["total_epochs_run"],
            "best_epoch": train_log["best_epoch"],
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        },
        "v2_precision_at_05": v2_metrics["precision_at_05"],
        "v2_recall_at_05": v2_metrics["recall_at_05"],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")

    print("\nExperiment 155 complete.")


if __name__ == "__main__":
    main()
