#!/usr/bin/env python3
"""Experiment 119: Adversarial GSM8K Dataset Generation.

**Researcher summary:**
    Reproduces the Apple Research methodology from arxiv 2410.05229 ("GSM-Symbolic:
    Understanding the Limitations of Mathematical Reasoning in Large Language
    Models"). Apple proved LLMs pattern-match math rather than reason by swapping
    numbers in GSM8K problems and adding irrelevant numeric distractors. Models
    dropped up to 65%. This experiment generates the four adversarial variants
    defined in that paper, saves the datasets, and spot-validates the answers.

**Detailed explanation for engineers:**
    The Apple paper used three key perturbation types:

    1. **Number swap (GSM-Symbolic)**: Same logical structure, different numeric
       values. A model that truly REASONS should adapt and give the new correct
       answer. A pattern-matching model will produce an answer close to the
       original problem's answer — revealing that it memorized the answer shape.

    2. **Irrelevant injection (GSM-NoOp)**: Add one sentence containing a number
       that is logically irrelevant to the computation. A reasoning model ignores
       it. A pattern-matching model treats it as a signal and incorporates it
       into its calculation, causing errors.

    3. **Combined**: Both perturbations applied simultaneously — the harshest
       adversarial condition.

    Implementation strategy:
    - Use synthetic GSM8K-style problems (same 15 templates as Exp 91/67) so
      we can deterministically recompute correct answers after number swaps.
      For each question i, we store the template index and two seeds: one for
      the control variant and one for the swapped variant. Re-running the same
      template with the swap seed produces structurally identical problems with
      different numeric values and a provably correct new answer.
    - Irrelevant sentences are drawn from a bank of 20+ templates. Each inserts
      a number that is plausible in context but has NO bearing on the answer.
    - Spot-check validation: for each of the 4 datasets, 10 items are selected,
      their answers are verified by rerunning the template arithmetic, and any
      discrepancy is flagged as a FAIL.

    Output: results/adversarial_gsm8k_data.json with keys:
    - "control": list of 200 items
    - "number_swapped": list of 200 items
    - "irrelevant_injected": list of 200 items
    - "combined": list of 200 items

    Each item has:
    - "id": integer index
    - "original_problem": original problem text
    - "perturbed_problem": adversarially modified text
    - "correct_answer": integer correct answer for the PERTURBED problem
    - "original_answer": integer correct answer for the ORIGINAL problem
    - "perturbation": one of "none" | "number_swap" | "irrelevant_injection" | "combined"
    - "template": template name for traceability
    - "control_seed": seed used for control generation
    - "swap_seed": seed used for swapped generation (same as control for non-swap)

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_119_adversarial_gsm8k.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-006
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "adversarial_gsm8k_data.json"

N_QUESTIONS = 200
SPOT_CHECK_N = 10
SEED_BASE = 119  # Experiment seed anchor


# ---------------------------------------------------------------------------
# Synthetic GSM8K-style problem templates (adapted from Exp 91/67)
# ---------------------------------------------------------------------------
# Each template function takes a random.Random instance and returns
# (question_text: str, answer: int). The same template called with a
# different Random instance produces the same STRUCTURE with different numbers
# and a different (but provably correct) answer.


def _tmpl_shopping(rng: random.Random) -> tuple[str, int]:
    """Shopping: buy items, apply discount, compute change."""
    n_shirts = rng.randint(2, 5)
    price = rng.randint(10, 30)
    n_pants = rng.randint(1, 3)
    price_p = rng.randint(25, 50)
    disc = rng.choice([10, 15, 20, 25])
    subtotal = n_shirts * price + n_pants * price_p
    discount = subtotal * disc // 100
    total = subtotal - discount
    budget = total + rng.randint(20, 80)
    change = budget - total
    return (
        f"Sarah wants to buy {n_shirts} shirts at ${price} each and "
        f"{n_pants} pairs of pants at ${price_p} each. The store offers "
        f"a {disc}% discount on the total. If Sarah has ${budget}, "
        f"how much change will she receive?",
        change,
    )


def _tmpl_cooking(rng: random.Random) -> tuple[str, int]:
    """Cooking: scale recipe ingredients."""
    base = rng.randint(2, 4)
    target = rng.randint(8, 16)
    eggs = rng.randint(2, 4)
    mult = target // base
    return (
        f"A recipe for {base} servings requires {eggs} eggs. "
        f"If Maria wants to make {target} servings, how many eggs does she need?",
        eggs * mult,
    )


def _tmpl_travel(rng: random.Random) -> tuple[str, int]:
    """Travel: distance, speed, fuel cost."""
    speed = rng.choice([40, 50, 60, 70, 80])
    hours = rng.randint(2, 5)
    mpg = rng.choice([20, 25, 30, 35])
    ppg = rng.randint(3, 5)
    dist = speed * hours
    gallons = dist // mpg
    return (
        f"Tom drives at {speed} mph for {hours} hours. His car gets "
        f"{mpg} miles per gallon, and gas costs ${ppg} per gallon. "
        f"How much does he spend on gas?",
        gallons * ppg,
    )


def _tmpl_savings(rng: random.Random) -> tuple[str, int]:
    """Savings: weekly deposits minus expense."""
    weeks = rng.randint(8, 20)
    weekly = rng.randint(15, 50)
    initial = rng.randint(50, 200)
    expense = rng.randint(30, 100)
    return (
        f"Emma starts with ${initial}. She saves ${weekly} every week "
        f"for {weeks} weeks, then spends ${expense} on a gift. "
        f"How much does she have?",
        initial + weeks * weekly - expense,
    )


def _tmpl_classroom(rng: random.Random) -> tuple[str, int]:
    """Classroom: supplies cost."""
    n = rng.randint(20, 35)
    pen = rng.randint(2, 5)
    pp = rng.randint(1, 3)
    nb = rng.randint(1, 3)
    np_ = rng.randint(3, 7)
    return (
        f"A teacher buys supplies for {n} students. Each student needs "
        f"{pen} pencils at ${pp} each and {nb} notebooks at ${np_} each. "
        f"What is the total cost?",
        n * (pen * pp + nb * np_),
    )


def _tmpl_garden(rng: random.Random) -> tuple[str, int]:
    """Garden: harvest minus consumption."""
    rows = rng.randint(4, 10)
    per_row = rng.randint(6, 15)
    per_plant = rng.randint(3, 8)
    eaten = rng.randint(5, 20)
    given = rng.randint(10, 30)
    total = rows * per_row * per_plant
    return (
        f"A garden has {rows} rows with {per_row} plants each producing "
        f"{per_plant} tomatoes. After eating {eaten} and giving {given} away, "
        f"how many remain?",
        total - eaten - given,
    )


def _tmpl_bakery(rng: random.Random) -> tuple[str, int]:
    """Bakery: batch production, sales, profit."""
    batches = rng.randint(3, 8)
    per_batch = rng.randint(12, 24)
    cost = rng.randint(5, 15)
    price = rng.randint(1, 3)
    unsold = rng.randint(5, 20)
    total = batches * per_batch
    sold = total - unsold
    return (
        f"A baker makes {batches} batches of {per_batch} cookies. "
        f"Each batch costs ${cost}. She sells cookies at ${price} each "
        f"but {unsold} go unsold. What is her profit?",
        sold * price - batches * cost,
    )


def _tmpl_library(rng: random.Random) -> tuple[str, int]:
    """Library: book inventory tracking."""
    init = rng.randint(100, 300)
    b1 = rng.randint(10, 30)
    r1 = rng.randint(5, 15)
    b2 = rng.randint(8, 25)
    r2 = rng.randint(10, 20)
    return (
        f"A library has {init} books. Monday: {b1} borrowed, {r1} returned. "
        f"Tuesday: {b2} borrowed, {r2} returned. How many now?",
        init - b1 + r1 - b2 + r2,
    )


def _tmpl_sports(rng: random.Random) -> tuple[str, int]:
    """Sports: total points across games."""
    games = rng.randint(4, 8)
    pts = [rng.randint(10, 35) for _ in range(games)]
    pts_str = ", ".join(str(p) for p in pts)
    return (
        f"A player scored {pts_str} points in {games} games. Total points?",
        sum(pts),
    )


def _tmpl_construction(rng: random.Random) -> tuple[str, int]:
    """Construction: workers x hours x days x rate."""
    w = rng.randint(3, 8)
    h = rng.randint(6, 10)
    d = rng.randint(3, 7)
    r = rng.randint(15, 40)
    return (
        f"{w} workers work {h} hours/day for {d} days, each laying "
        f"{r} bricks/hour. Total bricks?",
        w * h * d * r,
    )


def _tmpl_fundraiser(rng: random.Random) -> tuple[str, int]:
    """Fundraiser: ticket revenue + donation - expenses."""
    at = rng.randint(30, 80)
    ct = rng.randint(20, 50)
    ap = rng.randint(10, 25)
    cp = rng.randint(5, 12)
    don = rng.randint(50, 200)
    exp = rng.randint(100, 300)
    return (
        f"Fundraiser: {at} adult tickets at ${ap}, {ct} child tickets at ${cp}, "
        f"${don} donation, ${exp} expenses. Net raised?",
        at * ap + ct * cp + don - exp,
    )


def _tmpl_farm(rng: random.Random) -> tuple[str, int]:
    """Farm: egg production and sales."""
    ch = rng.randint(10, 30)
    epc = rng.randint(4, 7)
    wk = rng.randint(5, 10)
    eaten = rng.randint(5, 20)
    ppd = rng.randint(2, 5)
    total = ch * epc * wk
    for_sale = total - eaten
    dozens = for_sale // 12
    return (
        f"A farm has {ch} chickens laying {epc} eggs/week for {wk} weeks. "
        f"The farmer keeps {eaten} eggs and sells the rest at ${ppd}/dozen. Revenue?",
        dozens * ppd,
    )


def _tmpl_factory(rng: random.Random) -> tuple[str, int]:
    """Factory: production minus defects."""
    m = rng.randint(3, 8)
    upm = rng.randint(20, 50)
    h = rng.randint(4, 8)
    dp = rng.choice([5, 10, 15, 20])
    total = m * upm * h
    return (
        f"{m} machines producing {upm} units/hour for {h} hours. "
        f"{dp}% defective. How many good units?",
        total - total * dp // 100,
    )


def _tmpl_restaurant(rng: random.Random) -> tuple[str, int]:
    """Restaurant: bill split with tip."""
    n = rng.randint(3, 8)
    meals = [rng.randint(10, 30) for _ in range(n)]
    tip = rng.choice([15, 18, 20])
    sub = sum(meals)
    total = sub + sub * tip // 100
    return (
        f"{n} friends eat meals costing {', '.join(f'${m}' for m in meals)}. "
        f"They add {tip}% tip and split equally (round down). Per person?",
        total // n,
    )


def _tmpl_warehouse(rng: random.Random) -> tuple[str, int]:
    """Warehouse: inventory tracking over two days."""
    init = rng.randint(200, 500)
    r1 = rng.randint(50, 150)
    s1 = rng.randint(30, 100)
    r2 = rng.randint(40, 120)
    s2 = rng.randint(50, 130)
    dmg = rng.randint(5, 20)
    return (
        f"Warehouse starts with {init} items. Monday: +{r1} received, -{s1} shipped. "
        f"Tuesday: +{r2} received, -{s2} shipped. {dmg} items damaged. Usable items?",
        init + r1 - s1 + r2 - s2 - dmg,
    )


# All templates indexed by name for traceability in output JSON.
TEMPLATES: list[tuple[str, Callable[[random.Random], tuple[str, int]]]] = [
    ("shopping", _tmpl_shopping),
    ("cooking", _tmpl_cooking),
    ("travel", _tmpl_travel),
    ("savings", _tmpl_savings),
    ("classroom", _tmpl_classroom),
    ("garden", _tmpl_garden),
    ("bakery", _tmpl_bakery),
    ("library", _tmpl_library),
    ("sports", _tmpl_sports),
    ("construction", _tmpl_construction),
    ("fundraiser", _tmpl_fundraiser),
    ("farm", _tmpl_farm),
    ("factory", _tmpl_factory),
    ("restaurant", _tmpl_restaurant),
    ("warehouse", _tmpl_warehouse),
]


# ---------------------------------------------------------------------------
# Irrelevant sentence injection templates (Apple GSM-NoOp methodology)
# ---------------------------------------------------------------------------
# Each template contains a numeric placeholder {N} that will be filled with
# a random number. The sentence is semantically plausible but has NO effect
# on the answer to the problem. This tests whether models get distracted by
# irrelevant numeric information and try to incorporate it into their reasoning.

IRRELEVANT_TEMPLATES: list[str] = [
    "{N} of the items were slightly smaller than average.",
    "The store had been open for {N} years.",
    "She noticed {N} birds sitting outside the window.",
    "There were {N} other people waiting in the same line.",
    "The weather forecast predicted {N} degrees for that afternoon.",
    "He had {N} unread emails sitting in his inbox.",
    "The parking lot had {N} available spaces when they arrived.",
    "There were {N} chairs arranged in the waiting room.",
    "The building was constructed {N} years ago.",
    "The nearby park has {N} trees planted along the path.",
    "A survey showed that {N} percent of customers preferred the blue option.",
    "The clock on the wall showed that {N} minutes had already passed.",
    "There were {N} steps leading up to the main entrance.",
    "The newspaper reported that {N} people attended last year's event.",
    "A nearby sign indicated the speed limit was {N} miles per hour.",
    "The recipe book had {N} pages of color photographs.",
    "According to the forecast, there was a {N} percent chance of rain.",
    "The nearest bus stop was about {N} meters away.",
    "The office building had {N} floors in total.",
    "The local team had won {N} consecutive home games that season.",
    "The film had been playing for {N} minutes when they arrived.",
    "Her apartment was on the {N}th floor of the complex.",
]


def _generate_irrelevant_sentence(rng: random.Random, exclude_numbers: set[int]) -> str:
    """Generate one irrelevant sentence containing a number not in exclude_numbers.

    **Detailed explanation for engineers:**
        Selects a random template from IRRELEVANT_TEMPLATES and fills in the
        {N} placeholder with a number that does NOT appear in the original
        problem. This is important: if we accidentally inject a number that
        already appears in the problem AND happens to equal the answer, the
        model could be right for the wrong reason.

        The exclude set is built by parsing numbers from the problem text.
        We try up to 20 candidate values before giving up and just using
        a value that has no overlap with the exclude set.

    Args:
        rng: Random instance for reproducibility.
        exclude_numbers: Set of integers already present in the problem text.

    Returns:
        A complete sentence string with the {N} placeholder filled in.
    """
    template = rng.choice(IRRELEVANT_TEMPLATES)

    # Try to find a number not in exclude_numbers. Use a range that feels
    # natural for the chosen sentence context (2-99) with a few larger options.
    candidate_pool = list(range(2, 50)) + [60, 75, 100, 120, 150, 200]
    rng.shuffle(candidate_pool)

    chosen_n = candidate_pool[0]  # Default fallback
    for candidate in candidate_pool:
        if candidate not in exclude_numbers:
            chosen_n = candidate
            break

    return template.replace("{N}", str(chosen_n))


def _inject_irrelevant_sentence(problem: str, rng: random.Random) -> str:
    """Insert one irrelevant numeric sentence at a random position in the problem.

    **Detailed explanation for engineers:**
        Splits the problem into sentences using period boundaries, then inserts
        the irrelevant sentence at a random position (not the very last position,
        to avoid it looking like part of the question). This matches the
        Apple paper's approach of embedding distractors naturally in the problem
        narrative.

        We first collect all numbers present in the original problem so the
        injected number won't accidentally match an existing one.

    Args:
        problem: Original problem text.
        rng: Random instance for reproducibility.

    Returns:
        Modified problem text with one irrelevant sentence inserted.
    """
    # Collect numbers already in the problem to avoid injecting a matching one.
    existing_numbers: set[int] = set()
    for m in re.finditer(r"\b(\d+)\b", problem):
        try:
            existing_numbers.add(int(m.group(1)))
        except ValueError:
            pass

    irrelevant = _generate_irrelevant_sentence(rng, existing_numbers)

    # Split into sentences on period + space, then insert at a random non-final slot.
    # We use a simple sentence splitter that preserves the original spacing.
    # Split on ". " boundaries while keeping the delimiter.
    parts = re.split(r"(?<=\.)\s+", problem.strip())

    if len(parts) <= 1:
        # Single sentence: prepend the irrelevant sentence.
        return irrelevant + " " + problem
    else:
        # Insert at a random position (anywhere from index 1 to len-1 so it
        # doesn't appear at the very start or very end).
        insert_pos = rng.randint(1, max(1, len(parts) - 1))
        parts.insert(insert_pos, irrelevant)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Core adversarial variant generators
# ---------------------------------------------------------------------------


def number_swap(
    tmpl_fn: Callable[[random.Random], tuple[str, int]],
    swap_seed: int,
) -> tuple[str, int]:
    """Generate the number-swapped variant by re-running the template with a new seed.

    **Detailed explanation for engineers:**
        Because we use parameterized templates, a number-swap is equivalent to
        re-running the same template with a different random seed. The structure
        of the problem (what the question asks, the narrative context, the
        number of steps) is preserved, but every numeric value is different.

        This replicates the Apple paper's GSM-Symbolic methodology: same logical
        structure, genuinely different numbers with a provably correct new answer.
        The new answer is returned alongside the modified problem text so it can
        be used as ground truth when evaluating model responses.

    Args:
        tmpl_fn: The template function to re-run.
        swap_seed: A seed different from the control seed.

    Returns:
        Tuple of (perturbed_problem_text, new_correct_answer).
    """
    return tmpl_fn(random.Random(swap_seed))


def irrelevant_injection(
    problem: str,
    answer: int,
    inject_rng: random.Random,
) -> tuple[str, int]:
    """Inject one irrelevant numeric sentence; correct answer is unchanged.

    **Detailed explanation for engineers:**
        This replicates the Apple paper's GSM-NoOp (no-operation) methodology.
        A sentence containing a number that has NOTHING to do with the problem
        is inserted into the problem text. The CORRECT ANSWER DOES NOT CHANGE
        because the injected sentence adds no constraints to the problem.

        A model that truly reasons should ignore the irrelevant sentence.
        A model that pattern-matches on all visible numbers may incorporate the
        injected number into its calculation, causing it to produce a wrong answer.

    Args:
        problem: Original problem text.
        answer: Original correct answer (unchanged by this perturbation).
        inject_rng: Random instance for deterministic injection.

    Returns:
        Tuple of (perturbed_problem_with_injection, original_correct_answer).
    """
    perturbed = _inject_irrelevant_sentence(problem, inject_rng)
    return perturbed, answer  # Answer is unchanged — injection is irrelevant.


def combined_adversarial(
    tmpl_fn: Callable[[random.Random], tuple[str, int]],
    swap_seed: int,
    inject_rng: random.Random,
) -> tuple[str, int]:
    """Apply both number swap and irrelevant injection.

    **Detailed explanation for engineers:**
        First re-runs the template with a new seed (changing all numbers and
        the correct answer), then injects an irrelevant numeric sentence into
        the swapped problem. This is the most adversarial condition from the
        Apple paper — the model must simultaneously:
        1. Adapt to different numeric values (not pattern-match on memorized answers)
        2. Ignore the irrelevant numeric distractor

    Args:
        tmpl_fn: The template function for number swapping.
        swap_seed: Different seed for numeric variation.
        inject_rng: Random instance for sentence injection.

    Returns:
        Tuple of (fully_perturbed_problem, new_correct_answer).
    """
    swapped_problem, new_answer = tmpl_fn(random.Random(swap_seed))
    perturbed_problem, _ = irrelevant_injection(swapped_problem, new_answer, inject_rng)
    return perturbed_problem, new_answer


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_all_datasets(
    n: int = N_QUESTIONS, seed: int = SEED_BASE
) -> dict[str, list[dict[str, Any]]]:
    """Generate all four adversarial GSM8K dataset variants.

    **Detailed explanation for engineers:**
        Creates n problems for each of the 4 conditions (control, number-swapped,
        irrelevant-injected, combined) using deterministic random seeds derived
        from the base seed. All 4 variants for a given index i use the SAME
        template, so comparisons are valid — we're changing only the perturbation
        type, not the problem domain.

        Seed scheme per item i:
        - control_seed = seed + i * 1000          (for control and inject variants)
        - swap_seed    = seed + i * 1000 + 500     (for swap and combined variants)
        - inject_rng   = seeded with seed + i * 1000 + 250  (for injection variants)

        This ensures no seed collision across the four variants.

    Args:
        n: Number of questions per dataset.
        seed: Base random seed for reproducibility.

    Returns:
        Dict with keys "control", "number_swapped", "irrelevant_injected",
        "combined". Each value is a list of n dicts with fields: id,
        original_problem, perturbed_problem, correct_answer, original_answer,
        perturbation, template, control_seed, swap_seed.
    """
    control_items: list[dict[str, Any]] = []
    swapped_items: list[dict[str, Any]] = []
    injected_items: list[dict[str, Any]] = []
    combined_items: list[dict[str, Any]] = []

    n_templates = len(TEMPLATES)

    for i in range(n):
        tmpl_idx = i % n_templates
        tmpl_name, tmpl_fn = TEMPLATES[tmpl_idx]

        # Each item gets its own seed triplet to avoid cross-contamination.
        ctrl_seed = seed + i * 1000
        swap_seed = seed + i * 1000 + 500
        inj_seed = seed + i * 1000 + 250

        # Generate the control (original) version.
        orig_problem, orig_answer = tmpl_fn(random.Random(ctrl_seed))

        # --- Control variant: no perturbation ---
        control_items.append({
            "id": i,
            "original_problem": orig_problem,
            "perturbed_problem": orig_problem,   # Identical to original.
            "correct_answer": orig_answer,
            "original_answer": orig_answer,
            "perturbation": "none",
            "template": tmpl_name,
            "control_seed": ctrl_seed,
            "swap_seed": ctrl_seed,
        })

        # --- Number-swapped variant ---
        swap_problem, swap_answer = number_swap(tmpl_fn, swap_seed)
        swapped_items.append({
            "id": i,
            "original_problem": orig_problem,
            "perturbed_problem": swap_problem,
            "correct_answer": swap_answer,          # NEW answer after swapping.
            "original_answer": orig_answer,
            "perturbation": "number_swap",
            "template": tmpl_name,
            "control_seed": ctrl_seed,
            "swap_seed": swap_seed,
        })

        # --- Irrelevant injection variant ---
        inject_rng = random.Random(inj_seed)
        inj_problem, inj_answer = irrelevant_injection(orig_problem, orig_answer, inject_rng)
        injected_items.append({
            "id": i,
            "original_problem": orig_problem,
            "perturbed_problem": inj_problem,
            "correct_answer": inj_answer,           # SAME answer (injection is irrelevant).
            "original_answer": orig_answer,
            "perturbation": "irrelevant_injection",
            "template": tmpl_name,
            "control_seed": ctrl_seed,
            "swap_seed": ctrl_seed,
        })

        # --- Combined (swap + inject) variant ---
        comb_inject_rng = random.Random(inj_seed + 1)  # Different rng from inject-only.
        comb_problem, comb_answer = combined_adversarial(tmpl_fn, swap_seed, comb_inject_rng)
        combined_items.append({
            "id": i,
            "original_problem": orig_problem,
            "perturbed_problem": comb_problem,
            "correct_answer": comb_answer,          # Swapped answer + irrelevant sentence.
            "original_answer": orig_answer,
            "perturbation": "combined",
            "template": tmpl_name,
            "control_seed": ctrl_seed,
            "swap_seed": swap_seed,
        })

    return {
        "control": control_items,
        "number_swapped": swapped_items,
        "irrelevant_injected": injected_items,
        "combined": combined_items,
    }


# ---------------------------------------------------------------------------
# Spot-check validation
# ---------------------------------------------------------------------------


def spot_check_dataset(
    dataset: list[dict[str, Any]],
    name: str,
    n_check: int = SPOT_CHECK_N,
    seed: int = SEED_BASE + 9999,
) -> bool:
    """Spot-check n items from a dataset by regenerating their answers.

    **Detailed explanation for engineers:**
        For each sampled item, we rerun the template function with the recorded
        seed to independently recompute the expected answer. We then compare
        this to the stored correct_answer. Any mismatch indicates a bug in
        the generation logic (wrong seed recorded, wrong template called, etc.).

        For irrelevant_injection variants, we also verify that:
        1. The perturbed problem is LONGER than the original (sentence was added).
        2. The correct_answer matches the original_answer (injection is irrelevant).

        For number_swap variants, we also verify that:
        1. The perturbed problem text DIFFERS from the original.
        2. The correct_answer typically differs from the original_answer.
           (They could theoretically be equal by coincidence — we flag but don't fail.)

    Args:
        dataset: The list of question dicts to check.
        name: Human-readable name of the dataset (for print output).
        n_check: Number of items to spot-check.
        seed: Seed for selecting which items to check.

    Returns:
        True if all spot-checks pass, False if any fail.
    """
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n_check, len(dataset)))

    print(f"\n  Spot-checking {name} ({n_check} of {len(dataset)} items):")
    all_pass = True

    tmpl_map = {name: fn for name, fn in TEMPLATES}

    for idx in indices:
        item = dataset[idx]
        tmpl_name = item["template"]
        tmpl_fn = tmpl_map[tmpl_name]
        perturbation = item["perturbation"]
        stored_answer = item["correct_answer"]
        stored_orig_answer = item["original_answer"]

        # Recompute original answer.
        ctrl_seed = item["control_seed"]
        recomp_orig_problem, recomp_orig_answer = tmpl_fn(random.Random(ctrl_seed))

        # Recompute perturbed answer based on perturbation type.
        if perturbation == "none":
            recomp_answer = recomp_orig_answer
            text_diff_ok = item["perturbed_problem"] == item["original_problem"]
        elif perturbation == "number_swap":
            swap_seed = item["swap_seed"]
            _, recomp_answer = tmpl_fn(random.Random(swap_seed))
            text_diff_ok = item["perturbed_problem"] != item["original_problem"]
        elif perturbation == "irrelevant_injection":
            recomp_answer = recomp_orig_answer  # Answer unchanged by injection.
            text_diff_ok = len(item["perturbed_problem"]) > len(item["original_problem"])
        elif perturbation == "combined":
            swap_seed = item["swap_seed"]
            _, recomp_answer = tmpl_fn(random.Random(swap_seed))
            text_diff_ok = item["perturbed_problem"] != item["original_problem"]
        else:
            print(f"    [WARN] Unknown perturbation type: {perturbation}")
            recomp_answer = stored_answer
            text_diff_ok = True

        # Check original answer consistency.
        orig_ok = recomp_orig_answer == stored_orig_answer
        # Check perturbed answer consistency.
        answer_ok = recomp_answer == stored_answer
        # Both must pass.
        item_ok = orig_ok and answer_ok and text_diff_ok

        status = "PASS" if item_ok else "FAIL"
        if not item_ok:
            all_pass = False
            print(
                f"    [{status}] id={item['id']} template={tmpl_name} "
                f"perturbation={perturbation}"
            )
            print(f"           orig_stored={stored_orig_answer} orig_recomp={recomp_orig_answer}")
            print(f"           ans_stored={stored_answer}  ans_recomp={recomp_answer}")
            print(f"           text_diff_ok={text_diff_ok}")
        else:
            print(
                f"    [{status}] id={item['id']} template={tmpl_name} "
                f"answer={stored_answer} (orig={stored_orig_answer})"
            )

    return all_pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all four adversarial GSM8K dataset variants, validate, and save."""

    print("=" * 70)
    print("Experiment 119: Adversarial GSM8K Dataset Generation")
    print("Reproducing Apple arxiv 2410.05229 (GSM-Symbolic) methodology")
    print("=" * 70)

    # --- Step 1: Generate all 4 datasets (200 questions each) ---
    print(f"\n[Step 1] Generating {N_QUESTIONS} questions × 4 datasets...")
    datasets = generate_all_datasets(n=N_QUESTIONS, seed=SEED_BASE)

    for ds_name, items in datasets.items():
        n_items = len(items)
        unique_templates = {item["template"] for item in items}
        print(f"  {ds_name}: {n_items} items, {len(unique_templates)} templates used")

    # --- Step 2: Print dataset statistics ---
    print("\n[Step 2] Dataset statistics:")
    for ds_name, items in datasets.items():
        answers = [item["correct_answer"] for item in items]
        orig_answers = [item["original_answer"] for item in items]
        n_same = sum(1 for a, b in zip(answers, orig_answers) if a == b)
        n_diff = len(items) - n_same
        print(
            f"  {ds_name}: "
            f"min={min(answers)}, max={max(answers)}, "
            f"mean={sum(answers)/len(answers):.1f}, "
            f"answers_changed={n_diff}/{len(items)}"
        )

    # --- Step 3: Spot-check 10 from each dataset ---
    print("\n[Step 3] Running spot-check validation (10 per dataset)...")
    all_pass = True
    for ds_name, items in datasets.items():
        passed = spot_check_dataset(items, ds_name, n_check=SPOT_CHECK_N)
        if not passed:
            print(f"  *** VALIDATION FAILED for {ds_name} ***")
            all_pass = False

    if all_pass:
        print("\n  All spot-checks PASSED.")
    else:
        print("\n  *** SOME SPOT-CHECKS FAILED. See above for details. ***")

    # --- Step 4: Print example from each dataset ---
    print("\n[Step 4] Example items from each dataset:")
    for ds_name, items in datasets.items():
        item = items[3]  # Use index 3 for a consistent example.
        print(f"\n  === {ds_name.upper()} (template: {item['template']}) ===")
        print(f"  Original : {item['original_problem'][:120]}...")
        if item["perturbation"] != "none":
            print(f"  Perturbed: {item['perturbed_problem'][:120]}...")
        print(f"  Correct answer (perturbed): {item['correct_answer']}")
        print(f"  Original answer:            {item['original_answer']}")

    # --- Step 5: Save to results/adversarial_gsm8k_data.json ---
    print(f"\n[Step 5] Saving to {OUTPUT_PATH} ...")
    output = {
        "experiment": 119,
        "description": "Adversarial GSM8K (Apple 2410.05229 methodology reproduction)",
        "n_questions_per_dataset": N_QUESTIONS,
        "seed": SEED_BASE,
        "perturbations": {
            "none": "Control: unmodified synthetic GSM8K-style problem",
            "number_swap": "GSM-Symbolic: same template, different random seed = different numbers + new answer",
            "irrelevant_injection": "GSM-NoOp: irrelevant numeric sentence injected, answer unchanged",
            "combined": "Both number swap and irrelevant injection applied",
        },
        "datasets": datasets,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Saved {OUTPUT_PATH} ({size_mb:.2f} MB)")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for ds_name, items in datasets.items():
        print(f"  {ds_name}: {len(items)} questions")
    print(f"  Validation: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print(f"  Output: {OUTPUT_PATH}")
    print()
    if not all_pass:
        print("  ERROR: Spot-check validation failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
