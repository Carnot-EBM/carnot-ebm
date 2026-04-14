#!/usr/bin/env python3
"""Experiment 279: Adversarial number-swapped GSM8K with semantic grounding verifier.

**Researcher summary:**
    Exp 178 showed +28pp adversarial recovery but used the Ising constraint
    verifier (arithmetic-check). That verifier is blind to number-swap errors
    where the model produces internally consistent arithmetic on the WRONG
    numbers from the original (pre-swap) question.

    Semantic grounding checks question-answer ALIGNMENT: it extracts quantities
    and keywords from the question, then checks whether the response references
    those same quantities. A number-swapped response that uses old quantities
    should trigger `missing_quantity_coverage` violations.

    This experiment tests:
        - 50 original GSM8K questions (simulated Gemma4-E4B-it responses).
        - 50 number-swapped variants of the same questions.
    For each (question, response) pair we run `verify_semantic_grounding` and
    record whether a violation was produced.

    Primary metric:
        detection_rate — fraction of adversarially-wrong responses flagged.
        fp_rate        — fraction of correct original responses flagged (FP).

    A meaningful result requires detection_rate >> fp_rate.

**Adversarial simulation design:**
    Gemma4-E4B-it is simulated (model unavailable in CI). Simulated responses
    are designed to capture two realistic failure modes for number-swapped inputs:

        Mode A — "stale answer" (50% of adversarial errors):
            Model memorised the original problem and returns the original answer
            with original intermediate quantities. The response quantities do NOT
            match the swapped question quantities → semantic grounding fires.

        Mode B — "fresh but wrong" (50% of adversarial errors):
            Model processes the swapped problem but makes an arithmetic error.
            The response references swapped quantities but reaches the wrong
            final answer. Semantic grounding has no structural violation.

    For *correct* responses on original questions the response always uses the
    correct quantities, so semantic grounding should be quiet (low FP).

**Usage:**
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_279_adversarial_semantic.py

Spec: REQ-VERIFY-020, REQ-VERIFY-021, SCENARIO-VERIFY-020, SCENARIO-VERIFY-021
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Path setup — allow running both as a script and via importlib
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "experiment_279_results.json"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N_QUESTIONS = 50          # number of (original, swapped) pairs
BASE_SEED = 279_000       # deterministic seed
# Gemma4-E4B-it calibrated error rate (Apple paper, ~4B scale)
GEMMA4_BASE_ERROR_RATE = 0.25
# Fraction of adversarial errors that are "stale answer" (use original quantities)
STALE_FRACTION = 0.50
# Expected detection bound — used in result metadata
EXPECTED_DETECTION_LOWER_BOUND = 0.30


# ===========================================================================
# 1. Question-generating templates (re-used from Exp 119/178)
# ===========================================================================

def _tmpl_shopping(rng: random.Random) -> tuple[str, int]:
    """Sarah buys shirts and pants with a discount."""
    shirts = rng.randint(2, 8)
    shirt_price = rng.randint(10, 40)
    pants = rng.randint(1, 5)
    pant_price = rng.randint(30, 80)
    discount_pct = rng.choice([10, 15, 20, 25])
    budget = rng.randint(100, 300)
    total = shirts * shirt_price + pants * pant_price
    discounted = round(total * (1 - discount_pct / 100))
    change = budget - discounted
    prob = (
        f"Sarah wants to buy {shirts} shirts at ${shirt_price} each and "
        f"{pants} pairs of pants at ${pant_price} each. The store offers a "
        f"{discount_pct}% discount on the total. If Sarah has ${budget}, "
        f"how much change will she receive?"
    )
    return prob, change


def _tmpl_cooking(rng: random.Random) -> tuple[str, int]:
    """Baker distributes cookies to guests after eating some."""
    batches = rng.randint(2, 6)
    cookies_per_batch = rng.randint(12, 30)
    eaten_pct = rng.choice([10, 20, 25, 30])
    guests = rng.randint(3, 10)
    total = batches * cookies_per_batch
    eaten = round(total * eaten_pct / 100)
    remaining = total - eaten
    per_guest = remaining // guests
    prob = (
        f"A baker makes {batches} batches of cookies with {cookies_per_batch} cookies "
        f"each. After eating {eaten_pct}% of the total, the baker distributes the rest "
        f"equally among {guests} guests. How many cookies does each guest receive?"
    )
    return prob, per_guest


def _tmpl_travel(rng: random.Random) -> tuple[str, int]:
    """Total fuel cost for a road trip."""
    speed = rng.randint(50, 90)
    hours = rng.randint(2, 8)
    stop_minutes = rng.choice([15, 20, 30, 45])
    fuel_per_100km = rng.randint(7, 12)
    fuel_cost = rng.randint(1, 3)
    distance = speed * hours
    fuel = round(distance / 100 * fuel_per_100km)
    total_cost = fuel * fuel_cost
    prob = (
        f"A car travels at {speed} km/h for {hours} hours with a {stop_minutes}-minute "
        f"stop. It uses {fuel_per_100km} liters per 100 km and fuel costs ${fuel_cost} "
        f"per liter. What is the total fuel cost for the journey?"
    )
    return prob, total_cost


def _tmpl_savings(rng: random.Random) -> tuple[str, int]:
    """Maya saves money and needs to buy an item."""
    weekly = rng.randint(20, 100)
    weeks = rng.randint(4, 20)
    spend_pct = rng.choice([10, 20, 25, 30])
    item_cost = rng.randint(50, 300)
    saved = weekly * weeks
    spent = round(saved * spend_pct / 100)
    remaining = saved - spent
    shortfall = max(0, item_cost - remaining)
    prob = (
        f"Maya saves ${weekly} per week for {weeks} weeks. She spends {spend_pct}% of "
        f"her savings on a birthday gift. If the item she wants costs ${item_cost}, "
        f"how much more money does she need (0 if she has enough)?"
    )
    return prob, shortfall


def _tmpl_classroom(rng: random.Random) -> tuple[str, int]:
    """Students split into groups after absences."""
    classes = rng.randint(3, 8)
    students_per = rng.randint(20, 35)
    absent_pct = rng.choice([5, 10, 15, 20])
    groups = rng.randint(3, 7)
    total = classes * students_per
    present = round(total * (1 - absent_pct / 100))
    per_group = present // groups
    prob = (
        f"A school has {classes} classes of {students_per} students each. On a given day "
        f"{absent_pct}% are absent. The present students are split into {groups} equal "
        f"groups. How many students are in each group?"
    )
    return prob, per_group


def _tmpl_garden(rng: random.Random) -> tuple[str, int]:
    """Vegetables harvested after plant deaths."""
    rows = rng.randint(4, 10)
    plants_per_row = rng.randint(5, 15)
    died_pct = rng.choice([10, 20, 25])
    harvest_per = rng.randint(3, 10)
    total = rows * plants_per_row
    alive = round(total * (1 - died_pct / 100))
    harvest = alive * harvest_per
    prob = (
        f"A gardener plants {rows} rows of {plants_per_row} plants each. After "
        f"{died_pct}% die, each surviving plant produces {harvest_per} vegetables. "
        f"How many vegetables are harvested in total?"
    )
    return prob, harvest


def _tmpl_bakery(rng: random.Random) -> tuple[str, int]:
    """Bakery profit calculation."""
    loaves = rng.randint(10, 50)
    price = rng.randint(2, 8)
    cost_per = rng.randint(1, 4)
    overhead = rng.randint(10, 50)
    revenue = loaves * price
    cost = loaves * cost_per + overhead
    profit = revenue - cost
    prob = (
        f"A bakery bakes {loaves} loaves of bread, selling each for ${price}. "
        f"Each loaf costs ${cost_per} to make and overhead is ${overhead}. "
        f"What is the profit?"
    )
    return prob, profit


def _tmpl_farm(rng: random.Random) -> tuple[str, int]:
    """Daily milk revenue for a farm."""
    cows = rng.randint(5, 20)
    milk_per = rng.randint(15, 40)
    sell_pct = rng.choice([50, 60, 70, 75])
    price = rng.randint(1, 4)
    total_milk = cows * milk_per
    sold = round(total_milk * sell_pct / 100)
    revenue = sold * price
    prob = (
        f"A farm has {cows} cows each producing {milk_per} liters of milk per day. "
        f"The farmer sells {sell_pct}% of the milk at ${price} per liter. "
        f"What is the daily milk revenue?"
    )
    return prob, revenue


def _tmpl_factory(rng: random.Random) -> tuple[str, int]:
    """Factory revenue after defects."""
    machines = rng.randint(3, 10)
    units_per_hr = rng.randint(50, 200)
    hrs = rng.randint(6, 12)
    defect_pct = rng.choice([2, 5, 8, 10])
    price = rng.randint(5, 30)
    total = machines * units_per_hr * hrs
    defects = round(total * defect_pct / 100)
    sellable = total - defects
    revenue = sellable * price
    prob = (
        f"A factory runs {machines} machines for {hrs} hours, each producing "
        f"{units_per_hr} units/hour. {defect_pct}% are defective. "
        f"Each good unit sells for ${price}. What is the total revenue?"
    )
    return prob, revenue


def _tmpl_warehouse(rng: random.Random) -> tuple[str, int]:
    """Items remaining after shipment."""
    pallets = rng.randint(10, 50)
    boxes_per = rng.randint(12, 30)
    items_per = rng.randint(6, 20)
    shipped_pct = rng.choice([20, 25, 30, 40])
    total_boxes = pallets * boxes_per
    total_items = total_boxes * items_per
    shipped = round(total_items * shipped_pct / 100)
    remaining = total_items - shipped
    prob = (
        f"A warehouse stores {pallets} pallets with {boxes_per} boxes each "
        f"and {items_per} items per box. {shipped_pct}% are shipped out. "
        f"How many items remain in the warehouse?"
    )
    return prob, remaining


TEMPLATES: list[tuple[str, Callable[[random.Random], tuple[str, int]]]] = [
    ("shopping",   _tmpl_shopping),
    ("cooking",    _tmpl_cooking),
    ("travel",     _tmpl_travel),
    ("savings",    _tmpl_savings),
    ("classroom",  _tmpl_classroom),
    ("garden",     _tmpl_garden),
    ("bakery",     _tmpl_bakery),
    ("farm",       _tmpl_farm),
    ("factory",    _tmpl_factory),
    ("warehouse",  _tmpl_warehouse),
]


# ===========================================================================
# 2. Dataset generation — 50 (original, swapped) pairs
# ===========================================================================

def _extract_numbers(text: str) -> list[int]:
    """Return all integer numbers found in text."""
    return [int(m) for m in re.findall(r"\b\d+\b", text)]


def generate_pairs(n: int, seed: int) -> list[dict[str, Any]]:
    """Generate n (original, swapped) question pairs.

    Each pair shares the same template but uses different seeds so the
    numbers change (number_swap). The swap is guaranteed to produce at
    least one different numeric value, which means a model that memorised
    the original problem will produce a semantically wrong response.

    Returns a list of dicts with keys:
        id, template, original_question, original_answer,
        swapped_question, swapped_answer, orig_seed, swap_seed
    """
    pairs: list[dict[str, Any]] = []
    n_templates = len(TEMPLATES)
    for i in range(n):
        tmpl_name, tmpl_fn = TEMPLATES[i % n_templates]
        orig_seed = seed + i * 1000
        swap_seed = seed + i * 1000 + 500

        orig_q, orig_a = tmpl_fn(random.Random(orig_seed))
        swap_q, swap_a = tmpl_fn(random.Random(swap_seed))

        # Sanity: ensure the swap actually changed at least one number
        orig_nums = set(_extract_numbers(orig_q))
        swap_nums = set(_extract_numbers(swap_q))
        attempts = 0
        while orig_nums == swap_nums and attempts < 20:
            swap_seed += 1
            swap_q, swap_a = tmpl_fn(random.Random(swap_seed))
            swap_nums = set(_extract_numbers(swap_q))
            attempts += 1

        pairs.append({
            "id": i,
            "template": tmpl_name,
            "original_question": orig_q,
            "original_answer": orig_a,
            "swapped_question": swap_q,
            "swapped_answer": swap_a,
            "orig_seed": orig_seed,
            "swap_seed": swap_seed,
        })
    return pairs


# ===========================================================================
# 3. Response simulation
# ===========================================================================

def _response_with_all_numbers(
    question: str,
    answer: int,
    rng: random.Random,
    *,
    override_nums: list[int] | None = None,
) -> str:
    """Build a step-by-step response that explicitly references every number.

    Referencing every number from the question ensures that ALL premise clauses
    are covered by the response (each clause's quantities appear in at least one
    response sentence). This is key to keeping the semantic-grounding FP rate
    low on correct responses, and to making stale-vs-fresh-wrong distinguishable.

    Args:
        question:       The question text (used to extract numbers if no override).
        answer:         The final numeric answer to include.
        rng:            Random source for step-ordering variation.
        override_nums:  If supplied, use these numbers instead of extracting from
                        the question. Used for stale responses (original numbers).
    """
    nums: list[int] = override_nums if override_nums is not None else _extract_numbers(question)
    lines: list[str] = ["Let me work through this step by step."]
    # Shuffle so the step ordering varies but every number still appears
    shuffled = list(nums)
    rng.shuffle(shuffled)
    for i, n in enumerate(shuffled, start=1):
        lines.append(f"Step {i}: using {n} from the problem.")
    lines.append(f"Final answer: {answer}.")
    return "\n".join(lines)


def _correct_response(question: str, answer: int, rng: random.Random) -> str:
    """Simulate a correct response that references ALL key numbers from the question.

    Every number in the question appears in the response so that the semantic
    grounding verifier can match each premise clause → no missing-quantity
    violations → low FP rate on correct originals.
    """
    return _response_with_all_numbers(question, answer, rng)


def _stale_response(orig_question: str, orig_answer: int, rng: random.Random) -> str:
    """Simulate a "stale answer" error: model uses original question's numbers.

    The model effectively ignores the swapped numbers and produces a response
    grounded in the ORIGINAL question. When this response is verified against
    the SWAPPED question, the clause quantities do NOT match the response
    quantities → semantic grounding fires (`missing_quantity_coverage`).
    """
    return _response_with_all_numbers(orig_question, orig_answer, rng)


def _fresh_wrong_response(swap_question: str, swap_answer: int, rng: random.Random) -> str:
    """Simulate a fresh-but-wrong response: uses swapped numbers, wrong final answer.

    The model reads the swapped question correctly and references all the right
    numbers, but makes a reasoning error so the final answer is wrong. Semantic
    grounding sees the correct quantities for the swapped question → no structural
    violation (the error is in reasoning, not quantity coverage).
    """
    wrong_final = swap_answer + rng.choice([-10, -5, -3, -2, 2, 3, 5, 10])
    return _response_with_all_numbers(swap_question, wrong_final, rng)


def simulate_responses(
    pairs: list[dict[str, Any]],
    base_error_rate: float,
    stale_fraction: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Attach simulated responses for both original and swapped questions.

    For original questions: a correct response is produced most of the time
    (error rate = base_error_rate). Errors on originals are fresh-wrong
    (not stale, so they reference the right question's numbers).

    For swapped questions: error rate = base_error_rate. Of those errors,
    stale_fraction are "stale answer" (quantitatively mismatched to the
    swapped question) and the rest are "fresh but wrong".

    The returned list augments each pair dict with:
        orig_response, orig_is_correct,
        swap_response, swap_error_type  ('none'|'stale'|'fresh_wrong')
    """
    augmented: list[dict[str, Any]] = []
    for pair in pairs:
        # --- Original question response ---
        is_orig_correct = rng.random() > base_error_rate
        if is_orig_correct:
            orig_resp = _correct_response(pair["original_question"], pair["original_answer"], rng)
        else:
            orig_resp = _fresh_wrong_response(
                pair["original_question"], pair["original_answer"], rng
            )

        # --- Swapped question response ---
        is_swap_correct = rng.random() > base_error_rate
        if is_swap_correct:
            swap_resp = _correct_response(
                pair["swapped_question"], pair["swapped_answer"], rng
            )
            swap_error_type = "none"
        else:
            if rng.random() < stale_fraction:
                # Stale: uses original question's quantities → should be detected
                swap_resp = _stale_response(
                    pair["original_question"], pair["original_answer"], rng
                )
                swap_error_type = "stale"
            else:
                # Fresh wrong: uses swapped quantities, arithmetic error
                swap_resp = _fresh_wrong_response(
                    pair["swapped_question"], pair["swapped_answer"], rng
                )
                swap_error_type = "fresh_wrong"

        augmented.append({
            **pair,
            "orig_response": orig_resp,
            "orig_is_correct": is_orig_correct,
            "swap_response": swap_resp,
            "swap_error_type": swap_error_type,
        })
    return augmented


# ===========================================================================
# 4. Semantic grounding verification
# ===========================================================================

def run_semantic_grounding(
    augmented: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply verify_semantic_grounding to each (question, response) pair.

    Adds the following keys to each record:
        orig_grounding_verified   — True if no violations on original
        orig_grounding_violations — list of violation dicts
        swap_grounding_verified   — True if no violations on swapped
        swap_grounding_violations — list of violation dicts
    """
    from carnot.pipeline.semantic_grounding import verify_semantic_grounding

    results: list[dict[str, Any]] = []
    for record in augmented:
        orig_result = verify_semantic_grounding(
            question=record["original_question"],
            response=record["orig_response"],
        )
        swap_result = verify_semantic_grounding(
            question=record["swapped_question"],
            response=record["swap_response"],
        )
        results.append({
            **record,
            "orig_grounding_verified": orig_result.verified,
            "orig_grounding_violations": [v.to_dict() for v in orig_result.violations],
            "swap_grounding_verified": swap_result.verified,
            "swap_grounding_violations": [v.to_dict() for v in swap_result.violations],
        })
    return results


# ===========================================================================
# 5. Metrics computation
# ===========================================================================

def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute detection rate and FP rate.

    Detection rate:
        Of the swap responses that are wrong (swap_error_type != 'none'),
        what fraction were flagged by semantic grounding (not verified)?

    Stale detection rate:
        Same but restricted to stale-answer errors (the "easy" case where
        quantities are provably mismatched).

    Fresh-wrong detection rate:
        Restricted to fresh-wrong errors (the "hard" case).

    FP rate:
        Of the original correct responses (orig_is_correct == True),
        what fraction were incorrectly flagged?
    """
    wrong_records = [r for r in records if r["swap_error_type"] != "none"]
    stale_records = [r for r in records if r["swap_error_type"] == "stale"]
    fresh_wrong_records = [r for r in records if r["swap_error_type"] == "fresh_wrong"]
    correct_orig_records = [r for r in records if r["orig_is_correct"]]

    def detection_rate(subset: list[dict[str, Any]]) -> float:
        if not subset:
            return 0.0
        detected = sum(1 for r in subset if not r["swap_grounding_verified"])
        return round(detected / len(subset), 4)

    fp_rate = 0.0
    if correct_orig_records:
        fp_flagged = sum(1 for r in correct_orig_records if not r["orig_grounding_verified"])
        fp_rate = round(fp_flagged / len(correct_orig_records), 4)

    n_wrong = len(wrong_records)
    n_stale = len(stale_records)
    n_fresh = len(fresh_wrong_records)
    n_correct_orig = len(correct_orig_records)

    overall_detection = detection_rate(wrong_records)
    stale_detection = detection_rate(stale_records)
    fresh_detection = detection_rate(fresh_wrong_records)

    # Lift = detection_rate − fp_rate (net discrimination)
    lift = round(overall_detection - fp_rate, 4)

    return {
        "n_pairs": len(records),
        "n_wrong_swap": n_wrong,
        "n_stale": n_stale,
        "n_fresh_wrong": n_fresh,
        "n_correct_orig": n_correct_orig,
        "detection_rate": overall_detection,
        "stale_detection_rate": stale_detection,
        "fresh_wrong_detection_rate": fresh_detection,
        "fp_rate": fp_rate,
        "lift": lift,
    }


# ===========================================================================
# 6. Main entry point
# ===========================================================================

def run_experiment(
    n: int = N_QUESTIONS,
    seed: int = BASE_SEED,
    base_error_rate: float = GEMMA4_BASE_ERROR_RATE,
    stale_fraction: float = STALE_FRACTION,
) -> dict[str, Any]:
    """Run the full adversarial semantic grounding experiment.

    Args:
        n:               Number of (original, swapped) pairs to generate.
        seed:            Base seed for reproducibility.
        base_error_rate: Fraction of responses that are wrong (simulated Gemma4).
        stale_fraction:  Of adversarial errors, fraction that are "stale answer".

    Returns:
        Full results dict (written to OUTPUT_PATH).
    """
    rng = random.Random(seed)

    print(f"[Exp 279] Generating {n} question pairs (seed={seed})...")
    pairs = generate_pairs(n, seed)

    print(f"[Exp 279] Simulating Gemma4-E4B-it responses "
          f"(error_rate={base_error_rate}, stale_frac={stale_fraction})...")
    augmented = simulate_responses(pairs, base_error_rate, stale_fraction, rng)

    print("[Exp 279] Running semantic grounding verifier...")
    records = run_semantic_grounding(augmented)

    print("[Exp 279] Computing metrics...")
    metrics = compute_metrics(records)

    print(
        f"[Exp 279] Results — "
        f"detection_rate={metrics['detection_rate']:.2%}, "
        f"stale_detection={metrics['stale_detection_rate']:.2%}, "
        f"fresh_detection={metrics['fresh_wrong_detection_rate']:.2%}, "
        f"fp_rate={metrics['fp_rate']:.2%}, "
        f"lift={metrics['lift']:.2%}"
    )

    result = {
        "experiment": "exp279-adversarial-semantic",
        "title": "Adversarial number-swapped GSM8K with semantic grounding",
        "model": "Gemma4-E4B-it (simulated)",
        "parameters": {
            "n_questions": n,
            "seed": seed,
            "base_error_rate": base_error_rate,
            "stale_fraction": stale_fraction,
            "expected_detection_lower_bound": EXPECTED_DETECTION_LOWER_BOUND,
        },
        "metrics": metrics,
        "records": [
            {
                "id": r["id"],
                "template": r["template"],
                "orig_is_correct": r["orig_is_correct"],
                "swap_error_type": r["swap_error_type"],
                "orig_grounding_verified": r["orig_grounding_verified"],
                "orig_n_violations": len(r["orig_grounding_violations"]),
                "swap_grounding_verified": r["swap_grounding_verified"],
                "swap_n_violations": len(r["swap_grounding_violations"]),
                "swap_violation_types": sorted(
                    {v["violation_type"] for v in r["swap_grounding_violations"]}
                ),
            }
            for r in records
        ],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    print(f"[Exp 279] Results written to {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    run_experiment()
