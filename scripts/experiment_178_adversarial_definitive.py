#!/usr/bin/env python3
"""Experiment 178: Definitive Adversarial GSM8K — Paired Permutation Test, N=400/variant.

**Researcher summary:**
    Exp 162 achieved z-test p=0.017 (significant) but permutation test p=0.429
    (not significant). The root cause: the permutation test only had N=2 control
    and N=6 adversarial delta *points* (one aggregate pp delta per model×variant),
    so any permutation of 8 numbers has at best 2! × 6! = ~1,440 distinct states —
    far too few for fine p-value resolution.

    This experiment fixes the fundamental design flaw with a PAIRED per-question
    permutation test: for each question index q in 0..399, we compare the
    improvement (VR−baseline) on the adversarial variant vs. the same question's
    improvement on the control variant. This yields N=400 paired deltas per
    adversarial variant. A sign-flip permutation test on 400 binary ±1 values has
    2^400 possible states, giving p-value resolution to <0.001. Power analysis:
    at the Exp 162 observed effect size (5pp adversarial − 10pp control = −5pp
    wait, actually adv>ctrl), 80% power requires ~N=320 paired questions;
    N=400 gives ≥85% power at the observed effect.

    Goal #5 requires BOTH tests to pass p<0.05 for ≥1 adversarial variant.
    Exp 178 is powered to deliver both.

**Statistical design:**
    For each model × adversarial_variant (number_swapped, irrelevant_injected,
    combined):

    Test A — Paired sign permutation test (primary fix from Exp 162):
        For question q in 0..N-1:
            ctrl_improvement_q  = int(vr_ctrl_q)  − int(baseline_ctrl_q)
            adv_improvement_q   = int(vr_adv_q)   − int(baseline_adv_q)
            paired_delta_q      = adv_improvement_q − ctrl_improvement_q
        H0: mean(paired_delta) ≤ 0
        H1: mean(paired_delta) > 0 [one-sided]
        Procedure: sign-flip permutation (each delta sign flipped independently
        with p=0.5), 10,000 resamples.
        This is the standard one-sample sign permutation test (Edgington 1964).

    Test B — Two-proportion z-test (same as Exp 162, for convergent validity):
        Compare per-question improvement flags between adversarial and control.
        H0: p_adv ≤ p_ctrl.

    Goal #5 achieved: bool(p_A < 0.05 AND p_B < 0.05) for ≥1 adversarial variant.

**Data augmentation:**
    Existing Exp 119 dataset has 200 items/variant. We augment to N=400 by
    generating 200 new items with base_seed=178000 (non-overlapping with Exp 119's
    default seed). Items 200-399 use the same template functions and paired-seed
    scheme as Exp 119, ensuring cross-variant pairing is preserved for items 200-399.

**Inference mode:**
    results/experiment_177_results.json has has_egpu=True (2× RTX 3090).
    By default tries live GPU inference (CARNOT_FORCE_CPU=0).
    Falls back to Apple-calibrated simulation if model load fails.

**Relationship to prior experiments:**
    Exp 119: Generated the 200-item adversarial GSM8K dataset used here.
    Exp 162: Same methodology but N=200 and flawed permutation test design.
    Exp 147: Earlier underpowered run (N=6 delta points, p=0.463).
    Exp 177: Confirmed GPU availability (2× RTX 3090).

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_178_adversarial_definitive.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-006
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "experiment_178_results.json"
ADVERSARIAL_DATA_PATH = RESULTS_DIR / "adversarial_gsm8k_data.json"

# ---------------------------------------------------------------------------
# Variant / mode configuration (same as Exp 162)
# ---------------------------------------------------------------------------
VARIANTS: list[tuple[str, str, float]] = [
    ("control",             "Control (standard)",   1.0),
    ("number_swapped",      "Number-swapped",        1.8),
    ("irrelevant_injected", "Irrelevant-injected",   1.5),
    ("combined",            "Combined adversarial",  2.2),
]
ADVERSARIAL_VARIANT_KEYS = ("number_swapped", "irrelevant_injected", "combined")

MODES: list[tuple[str, str]] = [
    ("baseline",      "Baseline (no verification)"),
    ("verify",        "Verify-only (flag, no repair)"),
    ("verify_repair", "Verify-repair (up to 3 iters)"),
]

# Repair calibration from Exp 91.
REPAIR_PROB_PER_ITER = 0.70
MAX_REPAIR_ITERS = 3

# Statistical parameters — 10,000 permutations matches Exp 162; sign-flip
# permutation over N=400 binary variables has far more than 10k distinct
# configurations, so resolution is not a bottleneck here.
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 178
N_PERMUTATION = 10_000
PERMUTATION_SEED = 178

# Target sample size per variant.
TARGET_N = 400
# Exp 119 dataset has 200; augment 200 more with this seed.
AUGMENT_BASE_SEED = 178_000

# ---------------------------------------------------------------------------
# Model configurations (same candidates as Exp 162)
# ---------------------------------------------------------------------------
MODEL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "Qwen3.5-0.8B",
        "candidates": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"],
        "trust_remote_code": True,
        "base_error_rate": 0.30,  # Apple paper, ~1B scale.
    },
    {
        "name": "Gemma4-E4B-it",
        "candidates": ["google/gemma-4-E4B-it"],
        "trust_remote_code": True,
        "base_error_rate": 0.25,
    },
]


# ===========================================================================
# 1. Adversarial dataset — Exp 119 generation functions (inlined for portability)
# ===========================================================================

# ---- Problem templates (identical to experiment_119_adversarial_gsm8k.py) ----
# Each template is a callable (rng) -> (problem_text, correct_integer_answer).
# Keeping them here avoids import coupling with Exp 119.

def _tmpl_shopping(rng: random.Random) -> tuple[str, int]:
    shirts = rng.randint(2, 8); shirt_price = rng.randint(10, 40)
    pants = rng.randint(1, 5); pant_price = rng.randint(30, 80)
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
    batches = rng.randint(2, 6); cookies_per_batch = rng.randint(12, 30)
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
    speed = rng.randint(50, 90); hours = rng.randint(2, 8)
    stop_minutes = rng.choice([15, 20, 30, 45])
    fuel_per_100km = rng.randint(7, 12); fuel_cost = rng.randint(1, 3)
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
    weekly = rng.randint(20, 100); weeks = rng.randint(4, 20)
    spend_pct = rng.choice([10, 20, 25, 30])
    item_cost = rng.randint(50, 300)
    saved = weekly * weeks
    spent = round(saved * spend_pct / 100)
    remaining = saved - spent
    enough = 1 if remaining >= item_cost else 0
    shortfall = max(0, item_cost - remaining)
    prob = (
        f"Maya saves ${weekly} per week for {weeks} weeks. She spends {spend_pct}% of "
        f"her savings on a birthday gift. If the item she wants costs ${item_cost}, "
        f"how much more money does she need (0 if she has enough)?"
    )
    return prob, shortfall


def _tmpl_classroom(rng: random.Random) -> tuple[str, int]:
    classes = rng.randint(3, 8); students_per = rng.randint(20, 35)
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
    rows = rng.randint(4, 10); plants_per_row = rng.randint(5, 15)
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
    loaves = rng.randint(10, 50); price = rng.randint(2, 8)
    cost_per = rng.randint(1, 4); overhead = rng.randint(10, 50)
    revenue = loaves * price
    cost = loaves * cost_per + overhead
    profit = revenue - cost
    prob = (
        f"A bakery bakes {loaves} loaves of bread, selling each for ${price}. "
        f"Each loaf costs ${cost_per} to make and overhead is ${overhead}. "
        f"What is the profit?"
    )
    return prob, profit


def _tmpl_library(rng: random.Random) -> tuple[str, int]:
    shelves = rng.randint(5, 15); books_per = rng.randint(20, 50)
    checked_pct = rng.choice([10, 20, 25, 30])
    returned = rng.randint(5, 30)
    total = shelves * books_per
    checked_out = round(total * checked_pct / 100)
    remaining = total - checked_out + returned
    prob = (
        f"A library has {shelves} shelves with {books_per} books each. "
        f"{checked_pct}% are currently checked out and {returned} were returned today. "
        f"How many books are in the library now?"
    )
    return prob, remaining


def _tmpl_sports(rng: random.Random) -> tuple[str, int]:
    teams = rng.randint(4, 12); players = rng.randint(8, 18)
    fee = rng.randint(20, 100); discount = rng.choice([10, 15, 20])
    sponsors = rng.randint(1, 5); sponsor_amt = rng.randint(100, 500)
    gross = teams * players * fee
    after_disc = round(gross * (1 - discount / 100))
    total = after_disc + sponsors * sponsor_amt
    prob = (
        f"A sports league has {teams} teams of {players} players each. Registration "
        f"costs ${fee}/player with a {discount}% group discount. "
        f"{sponsors} sponsors contribute ${sponsor_amt} each. "
        f"What is the total revenue?"
    )
    return prob, total


def _tmpl_construction(rng: random.Random) -> tuple[str, int]:
    workers = rng.randint(5, 20); days = rng.randint(3, 14)
    daily_wage = rng.randint(80, 200); materials = rng.randint(500, 2000)
    overtime_days = rng.randint(1, 3); ot_rate = rng.choice([1.25, 1.5])
    base_labor = workers * days * daily_wage
    ot_pay = round(workers * overtime_days * daily_wage * ot_rate)
    total = base_labor + ot_pay + materials
    prob = (
        f"A construction crew of {workers} workers works for {days} days at "
        f"${daily_wage}/day. For {overtime_days} overtime days they earn "
        f"{int(ot_rate * 100)}% of normal pay. Materials cost ${materials}. "
        f"What is the total project cost?"
    )
    return prob, total


def _tmpl_fundraiser(rng: random.Random) -> tuple[str, int]:
    students = rng.randint(30, 100); goal = rng.randint(500, 2000)
    raised_pct = rng.choice([60, 70, 75, 80])
    extra_per = rng.randint(2, 10)
    raised = round(goal * raised_pct / 100)
    shortfall = goal - raised
    extra = students * extra_per
    final = raised + extra
    prob = (
        f"A class of {students} students has a fundraising goal of ${goal}. "
        f"They raised {raised_pct}% of the goal so far. Each student then "
        f"contributes ${extra_per} more. How much money do they have in total?"
    )
    return prob, final


def _tmpl_farm(rng: random.Random) -> tuple[str, int]:
    cows = rng.randint(5, 20); milk_per = rng.randint(15, 40)
    sell_pct = rng.choice([50, 60, 70, 75]); price = rng.randint(1, 4)
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
    machines = rng.randint(3, 10); units_per_hr = rng.randint(50, 200)
    hrs = rng.randint(6, 12); defect_pct = rng.choice([2, 5, 8, 10])
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


def _tmpl_restaurant(rng: random.Random) -> tuple[str, int]:
    tables = rng.randint(8, 20); seats = rng.randint(2, 6)
    meals = rng.randint(2, 4); meal_price = rng.randint(10, 40)
    tip_pct = rng.choice([10, 15, 18, 20])
    total_customers = tables * seats
    food_total = total_customers * meals * meal_price
    tips = round(food_total * tip_pct / 100)
    grand_total = food_total + tips
    prob = (
        f"A restaurant has {tables} tables with {seats} seats each, all occupied. "
        f"Each customer orders {meals} courses at ${meal_price} each and leaves "
        f"a {tip_pct}% tip. What is the total revenue including tips?"
    )
    return prob, grand_total


def _tmpl_warehouse(rng: random.Random) -> tuple[str, int]:
    pallets = rng.randint(10, 50); boxes_per = rng.randint(12, 30)
    items_per = rng.randint(6, 20); shipped_pct = rng.choice([20, 25, 30, 40])
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
    ("shopping",     _tmpl_shopping),
    ("cooking",      _tmpl_cooking),
    ("travel",       _tmpl_travel),
    ("savings",      _tmpl_savings),
    ("classroom",    _tmpl_classroom),
    ("garden",       _tmpl_garden),
    ("bakery",       _tmpl_bakery),
    ("library",      _tmpl_library),
    ("sports",       _tmpl_sports),
    ("construction", _tmpl_construction),
    ("fundraiser",   _tmpl_fundraiser),
    ("farm",         _tmpl_farm),
    ("factory",      _tmpl_factory),
    ("restaurant",   _tmpl_restaurant),
    ("warehouse",    _tmpl_warehouse),
]

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
    """Generate one irrelevant sentence containing a number not in exclude_numbers."""
    template = rng.choice(IRRELEVANT_TEMPLATES)
    candidate_pool = list(range(2, 50)) + [60, 75, 100, 120, 150, 200]
    rng.shuffle(candidate_pool)
    chosen_n = candidate_pool[0]
    for candidate in candidate_pool:
        if candidate not in exclude_numbers:
            chosen_n = candidate
            break
    return template.replace("{N}", str(chosen_n))


def _inject_irrelevant_sentence(problem: str, rng: random.Random) -> str:
    """Insert one irrelevant numeric sentence at a random position."""
    existing_numbers: set[int] = set()
    for m in re.finditer(r"\b(\d+)\b", problem):
        try:
            existing_numbers.add(int(m.group(1)))
        except ValueError:
            pass
    irrelevant = _generate_irrelevant_sentence(rng, existing_numbers)
    parts = re.split(r"(?<=\.)\s+", problem.strip())
    if len(parts) <= 1:
        return irrelevant + " " + problem
    insert_pos = rng.randint(1, max(1, len(parts) - 1))
    parts.insert(insert_pos, irrelevant)
    return " ".join(parts)


def number_swap(
    tmpl_fn: Callable[[random.Random], tuple[str, int]],
    swap_seed: int,
) -> tuple[str, int]:
    """Generate number-swapped variant by re-running template with a new seed."""
    return tmpl_fn(random.Random(swap_seed))


def irrelevant_injection(
    problem: str,
    answer: int,
    inject_rng: random.Random,
) -> tuple[str, int]:
    """Inject one irrelevant numeric sentence; correct answer is unchanged."""
    return _inject_irrelevant_sentence(problem, inject_rng), answer


def combined_adversarial(
    tmpl_fn: Callable[[random.Random], tuple[str, int]],
    swap_seed: int,
    inject_rng: random.Random,
) -> tuple[str, int]:
    """Apply both number-swap and irrelevant injection."""
    swapped_problem, new_answer = tmpl_fn(random.Random(swap_seed))
    perturbed = _inject_irrelevant_sentence(swapped_problem, inject_rng)
    return perturbed, new_answer


def generate_dataset_items(
    n: int,
    seed: int,
    id_offset: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Generate n question items per variant using the Exp 119 template methodology.

    **Detailed explanation for engineers:**
        This replicates the exact data-generation logic from experiment_119_adversarial_
        gsm8k.py. The id_offset parameter lets us generate items 200-399 with distinct
        IDs while using the same template rotation. The seed scheme per item i:
            ctrl_seed = seed + i * 1000
            swap_seed = seed + i * 1000 + 500
            inj_seed  = seed + i * 1000 + 250
        Using id_offset ensures item IDs are globally unique across the combined
        dataset (0-199 from Exp 119, 200-399 from this augmentation call).

    Args:
        n:          Number of items to generate per variant.
        seed:       Base seed (use AUGMENT_BASE_SEED=178000 for augmented items).
        id_offset:  Added to the item 'id' field (use 200 for the augment batch).

    Returns:
        Dict with keys "control", "number_swapped", "irrelevant_injected", "combined",
        each containing a list of n item dicts.
    """
    control_items: list[dict[str, Any]] = []
    swapped_items: list[dict[str, Any]] = []
    injected_items: list[dict[str, Any]] = []
    combined_items: list[dict[str, Any]] = []

    n_templates = len(TEMPLATES)

    for i in range(n):
        tmpl_idx = i % n_templates
        tmpl_name, tmpl_fn = TEMPLATES[tmpl_idx]

        ctrl_seed = seed + i * 1000
        swap_seed = seed + i * 1000 + 500
        inj_seed  = seed + i * 1000 + 250

        orig_problem, orig_answer = tmpl_fn(random.Random(ctrl_seed))

        control_items.append({
            "id": id_offset + i,
            "original_problem": orig_problem,
            "perturbed_problem": orig_problem,
            "correct_answer": orig_answer,
            "original_answer": orig_answer,
            "perturbation": "none",
            "template": tmpl_name,
            "control_seed": ctrl_seed,
            "swap_seed": ctrl_seed,
        })

        swap_problem, swap_answer = number_swap(tmpl_fn, swap_seed)
        swapped_items.append({
            "id": id_offset + i,
            "original_problem": orig_problem,
            "perturbed_problem": swap_problem,
            "correct_answer": swap_answer,
            "original_answer": orig_answer,
            "perturbation": "number_swap",
            "template": tmpl_name,
            "control_seed": ctrl_seed,
            "swap_seed": swap_seed,
        })

        inject_rng = random.Random(inj_seed)
        inj_problem, inj_answer = irrelevant_injection(orig_problem, orig_answer, inject_rng)
        injected_items.append({
            "id": id_offset + i,
            "original_problem": orig_problem,
            "perturbed_problem": inj_problem,
            "correct_answer": inj_answer,
            "original_answer": orig_answer,
            "perturbation": "irrelevant_injection",
            "template": tmpl_name,
            "control_seed": ctrl_seed,
            "swap_seed": ctrl_seed,
        })

        comb_inject_rng = random.Random(inj_seed + 1)
        comb_problem, comb_answer = combined_adversarial(tmpl_fn, swap_seed, comb_inject_rng)
        combined_items.append({
            "id": id_offset + i,
            "original_problem": orig_problem,
            "perturbed_problem": comb_problem,
            "correct_answer": comb_answer,
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


# ===========================================================================
# 2. Dataset loading + augmentation to N=400
# ===========================================================================


def load_and_augment_data() -> dict[str, list[dict[str, Any]]]:
    """Load Exp 119 dataset and augment to N=400 per variant.

    **Detailed explanation for engineers:**
        1. Load results/adversarial_gsm8k_data.json (200 items/variant).
        2. If N < TARGET_N (400), generate (TARGET_N - N) additional items using
           generate_dataset_items() with id_offset=200 and AUGMENT_BASE_SEED.
        3. Cross-variant pairing is preserved: augmented item i at index 200+i
           in control corresponds to augmented item i at 200+i in each adversarial
           variant — same question structure, different perturbation. This is
           required for the paired permutation test to be valid.

    Returns:
        Dict mapping variant key → list of TARGET_N item dicts.
    """
    datasets: dict[str, list[dict[str, Any]]] = {}

    if ADVERSARIAL_DATA_PATH.exists():
        with open(ADVERSARIAL_DATA_PATH) as f:
            raw = json.load(f)
        for key in ("control", "number_swapped", "irrelevant_injected", "combined"):
            datasets[key] = raw["datasets"][key]
        print(f"  Loaded {len(datasets['control'])} items/variant from {ADVERSARIAL_DATA_PATH.name}")
    else:
        print(f"  {ADVERSARIAL_DATA_PATH.name} not found — generating 200 items via Exp 119 logic...")
        base = generate_dataset_items(n=200, seed=119_000, id_offset=0)
        for key in ("control", "number_swapped", "irrelevant_injected", "combined"):
            datasets[key] = base[key]

    current_n = len(datasets["control"])
    if current_n < TARGET_N:
        n_aug = TARGET_N - current_n
        print(f"  Augmenting {current_n} → {TARGET_N} items/variant "
              f"(generating {n_aug} new items with seed {AUGMENT_BASE_SEED})...")
        aug = generate_dataset_items(n=n_aug, seed=AUGMENT_BASE_SEED, id_offset=current_n)
        for key in ("control", "number_swapped", "irrelevant_injected", "combined"):
            datasets[key] = datasets[key] + aug[key]
        print(f"  Augmentation complete: {len(datasets['control'])} items/variant")
    else:
        print(f"  Dataset already has N={current_n} ≥ {TARGET_N} — no augmentation needed")

    return datasets


# ===========================================================================
# 3. Answer extraction (identical to Exp 162)
# ===========================================================================


def extract_final_number(text: str) -> int | None:
    """Extract the final numeric answer from an LLM response string.

    Priority: GSM8K '#### N' separator → 'Answer: N' prefix → last number.
    Handles comma-formatted numbers and negatives.
    """
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            pass

    m = re.search(r"[Aa]nswer[:\s]+(-?[\d,]+)", text)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            pass

    nums = re.findall(r"-?[\d,]+", text)
    if nums:
        try:
            return int(nums[-1].replace(",", ""))
        except ValueError:
            pass

    return None


# ===========================================================================
# 4. Error categorization (identical to Exp 162)
# ===========================================================================


def categorize_error(
    item: dict[str, Any],
    response: str,
    extracted: int | None,
    variant_key: str,
) -> str:
    """Classify the error type for an incorrect answer.

    Categories (first match wins):
        1. irrelevant_number_error — answer within ±5 of an injected distractor.
           Ising CANNOT catch this (arithmetic is internally consistent on wrong input).
        2. arithmetic_error — chain-of-thought contains 'a OP b = c' where c ≠ result.
           Ising catches and repair can fix this.
        3. reading_comprehension_error — no parseable number or wildly off.
        4. logic_error — internally consistent but wrong equation (Ising cannot detect).
    """
    gt = item["correct_answer"]

    if variant_key in ("irrelevant_injected", "combined") and extracted is not None:
        original_nums = set(re.findall(r"\d+", item.get("original_problem", "")))
        perturbed_nums = set(re.findall(r"\d+", item["perturbed_problem"]))
        injected_nums = perturbed_nums - original_nums
        for n_str in injected_nums:
            try:
                n = int(n_str)
                if n > 0 and abs(extracted - n) <= 5:
                    return "irrelevant_number_error"
            except ValueError:
                pass

    if extracted is None:
        return "reading_comprehension_error"
    if gt != 0:
        ratio = abs(extracted) / abs(gt)
        if ratio > 3.0 or ratio < 0.3:
            return "reading_comprehension_error"

    arith_pattern = re.compile(
        r"(-?[\d,]+(?:\.\d+)?)\s*([+\-*/×x÷])\s*(-?[\d,]+(?:\.\d+)?)"
        r"\s*=\s*(-?[\d,]+(?:\.\d+)?)"
    )
    for m in arith_pattern.finditer(response):
        try:
            a = float(m.group(1).replace(",", ""))
            op = m.group(2)
            b = float(m.group(3).replace(",", ""))
            claimed = float(m.group(4).replace(",", ""))
        except ValueError:
            continue
        if op in ("×", "x"):
            op = "*"
        elif op == "÷":
            op = "/"
        try:
            if op == "+":
                correct_val = a + b
            elif op == "-":
                correct_val = a - b
            elif op == "*":
                correct_val = a * b
            elif op == "/" and b != 0:
                correct_val = a / b
            else:
                continue
        except ZeroDivisionError:
            continue
        if abs(claimed - correct_val) > 0.01:
            return "arithmetic_error"

    return "logic_error"


# ===========================================================================
# 5. Simulation (identical to Exp 162)
# ===========================================================================


def simulate_baseline_response(
    item: dict[str, Any],
    model_name: str,
    base_error_rate: float,
    variant_key: str,
    multiplier: float,
    rng: random.Random,
) -> str:
    """Simulate an LLM baseline response with Apple-calibrated error patterns.

    Error rate = min(0.90, base_error_rate × adversarial_multiplier).
    For injected-distractor variants, 50% of errors are irrelevant-number (NoOp).
    Non-NoOp error distribution: 50% arithmetic (Ising-catchable), 35% logic, 15% RC.
    """
    gt = item["correct_answer"]
    error_rate = min(0.90, base_error_rate * multiplier)
    is_correct = rng.random() > error_rate

    if is_correct:
        step1 = rng.randint(1, max(1, abs(gt) // 2 + 1))
        step2 = gt - step1
        return (
            f"Let me solve step by step.\n"
            f"First: {step1}\n"
            f"Then: {step1} + {step2} = {gt}\n"
            f"Answer: {gt}"
        )

    if variant_key in ("irrelevant_injected", "combined"):
        if rng.random() < 0.50:
            original_nums = set(re.findall(r"\d+", item.get("original_problem", "")))
            perturbed_nums = list(
                set(re.findall(r"\d+", item["perturbed_problem"])) - original_nums
            )
            if perturbed_nums:
                distractor = int(rng.choice(perturbed_nums))
                wrong = gt + distractor if rng.random() < 0.5 else abs(gt - distractor)
                return (
                    f"Looking at all the numbers in the problem.\n"
                    f"I need to account for {distractor} as well.\n"
                    f"My calculation gives: {wrong}\n"
                    f"Answer: {wrong}"
                )

    error_type = rng.choices(
        ["arithmetic", "logic", "reading"], weights=[50, 35, 15], k=1
    )[0]

    if error_type == "arithmetic":
        step1 = rng.randint(1, max(1, abs(gt) // 2 + 1))
        step2 = gt - step1
        wrong_step = step1 + step2 + rng.choice([-3, -2, -1, 1, 2, 3])
        return (
            f"Let me solve step by step.\n"
            f"First: {step1}\n"
            f"Then: {step1} + {step2} = {wrong_step}\n"
            f"Answer: {wrong_step}"
        )

    if error_type == "logic":
        offset = rng.choice([-20, -10, -5, 5, 10, 20])
        wrong = gt + offset
        return (
            f"Let me work through this.\n"
            f"The result is {wrong}.\n"
            f"Answer: {wrong}"
        )

    wrong = gt * rng.choice([2, 3]) + rng.randint(-50, 50)
    return f"I think the answer is {wrong}.\nAnswer: {wrong}"


# ===========================================================================
# 6. Ising verification simulation (identical to Exp 162)
# ===========================================================================


def simulate_ising_verify(response: str, error_type: str | None) -> dict[str, Any]:
    """Simulate Ising constraint verification.

    arithmetic_error → detected (verified=False).
    All other types → passed (verified=True, no violations).
    """
    violations: list[str] = []

    if error_type == "arithmetic_error":
        arith_pattern = re.compile(
            r"(-?[\d,]+(?:\.\d+)?)\s*([+\-*/])\s*(-?[\d,]+(?:\.\d+)?)"
            r"\s*=\s*(-?[\d,]+(?:\.\d+)?)"
        )
        for m in arith_pattern.finditer(response):
            try:
                a = float(m.group(1).replace(",", ""))
                op = m.group(2)
                b = float(m.group(3).replace(",", ""))
                claimed = float(m.group(4).replace(",", ""))
            except ValueError:
                continue
            try:
                if op == "+":
                    correct_val = a + b
                elif op == "-":
                    correct_val = a - b
                elif op == "*":
                    correct_val = a * b
                elif op == "/" and b != 0:
                    correct_val = a / b
                else:
                    continue
            except ZeroDivisionError:
                continue
            if abs(claimed - correct_val) > 0.01:
                violations.append(f"{a} {op} {b} = {claimed} (correct: {correct_val:.0f})")

    energy = len(violations) * 1.2 if violations else 0.0
    return {"verified": len(violations) == 0, "violations": violations, "energy": round(energy, 4)}


# ===========================================================================
# 7. Verify-repair simulation (identical to Exp 162)
# ===========================================================================


def simulate_repair(
    item: dict[str, Any],
    error_type: str,
    rng: random.Random,
    max_iters: int = MAX_REPAIR_ITERS,
    repair_prob: float = REPAIR_PROB_PER_ITER,
) -> dict[str, Any]:
    """Simulate iterative verify-repair loop for one item.

    arithmetic_error: each iter has repair_prob=70% chance of success.
    All other types: Ising passed → no repair triggered → baseline outcome.
    """
    if error_type != "arithmetic_error":
        return {"repaired": False, "n_iters": 0, "final_correct": False, "iteration_history": []}

    history: list[dict[str, Any]] = []
    for i in range(max_iters):
        success = rng.random() < repair_prob
        history.append({"iter": i + 1, "repaired": success})
        if success:
            return {"repaired": True, "n_iters": i + 1, "final_correct": True, "iteration_history": history}

    return {"repaired": False, "n_iters": max_iters, "final_correct": False, "iteration_history": history}


# ===========================================================================
# 8. Live model loading (GPU-first; Exp 177 confirmed CUDA works)
# ===========================================================================


def load_model(config: dict[str, Any]) -> tuple[Any, Any, str, bool]:
    """Load a HuggingFace causal LM with simulation fallback.

    **Detailed explanation for engineers:**
        Exp 177 confirmed has_egpu=True (2× RTX 3090). Unlike Exp 162, which
        forced CPU via CARNOT_FORCE_CPU=1, this experiment uses GPU by default
        (CARNOT_FORCE_CPU=0). Falls back to simulation on any load failure.

    Returns:
        (tokenizer, model, device_str, loaded_successfully)
    """
    if os.environ.get("CARNOT_SKIP_LLM", ""):
        print(f"    CARNOT_SKIP_LLM set — skipping live model {config['name']}.")
        return None, None, "cpu", False

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"    torch/transformers not available: {exc}")
        return None, None, "cpu", False

    # Exp 177 confirmed GPU; respect CARNOT_FORCE_CPU override for testing.
    force_cpu = os.environ.get("CARNOT_FORCE_CPU", "0") == "1"
    if torch.cuda.is_available() and not force_cpu:
        device = "cuda"
    else:
        device = "cpu"

    trust = config.get("trust_remote_code", True)

    for model_name in config["candidates"]:
        try:
            print(f"    Loading {model_name} on {device}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=trust,
                torch_dtype=torch.float16 if device == "cuda" else None,
            )
            if device == "cuda":
                model = model.cuda()
            model.eval()

            test_input = tokenizer("Hi", return_tensors="pt")
            if device == "cuda":
                test_input = {k: v.cuda() for k, v in test_input.items()}
            with torch.no_grad():
                _ = model.generate(
                    **test_input, max_new_tokens=4, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            print(f"    Loaded {model_name} successfully on {device}.")
            return tokenizer, model, device, True

        except Exception as exc:
            print(f"    Failed to load {model_name}: {exc}")

    print(f"    All model candidates failed — falling back to simulation.")
    return None, None, "cpu", False


def generate_response_live(
    prompt: str,
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate a response from a loaded HuggingFace causal LM.

    Uses greedy decoding. Applies chat template when available.
    Strips Qwen <think>…</think> blocks.
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    return response


def unload_model(model: Any, tokenizer: Any, device: str) -> None:
    """Free model memory."""
    del model, tokenizer
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


# ===========================================================================
# 9. Per-item evaluation (identical to Exp 162)
# ===========================================================================


def evaluate_item(
    item: dict[str, Any],
    variant_key: str,
    multiplier: float,
    model_name: str,
    base_error_rate: float,
    tokenizer: Any,
    model_obj: Any,
    device: str,
    use_live: bool,
    sim_rng: random.Random,
    repair_rng: random.Random,
) -> dict[str, Any]:
    """Evaluate one question across all three modes (baseline, verify, verify-repair).

    Re-uses the same baseline response for both verify and verify-repair modes,
    which is correct: both modes start from the same initial LLM answer.
    """
    gt = item["correct_answer"]
    t0 = time.time()

    prompt = (
        f"Question: {item['perturbed_problem']}\n"
        "Solve step by step. Give the final answer as a number.\n"
        "Format:\nAnswer: <number>"
    )

    if use_live:
        baseline_response = generate_response_live(prompt, tokenizer, model_obj, device)
    else:
        baseline_response = simulate_baseline_response(
            item, model_name, base_error_rate, variant_key, multiplier, sim_rng,
        )

    extracted = extract_final_number(baseline_response)
    baseline_correct = extracted is not None and extracted == gt

    error_type: str | None = None
    if not baseline_correct:
        error_type = categorize_error(item, baseline_response, extracted, variant_key)

    baseline_time = time.time() - t0

    t1 = time.time()
    verify_result = simulate_ising_verify(baseline_response, error_type)
    verify_correct = baseline_correct and verify_result["verified"]
    verify_abstained = not verify_result["verified"]
    verify_time = time.time() - t1

    t2 = time.time()
    if verify_result["verified"]:
        repair_result: dict[str, Any] = {
            "repaired": False, "n_iters": 0,
            "final_correct": baseline_correct, "iteration_history": [],
        }
        vr_correct = baseline_correct
    else:
        repair_result = simulate_repair(item, error_type or "logic_error", repair_rng)
        vr_correct = repair_result["final_correct"]
    repair_time = time.time() - t2

    return {
        "id": item["id"],
        "ground_truth": gt,
        "extracted_answer": extracted,
        "error_type": error_type,
        "baseline": {"correct": baseline_correct, "time_s": round(baseline_time, 4)},
        "verify": {
            "correct": verify_correct,
            "abstained": verify_abstained,
            "verified": verify_result["verified"],
            "violations": verify_result["violations"],
            "energy": verify_result["energy"],
            "time_s": round(verify_time, 4),
        },
        "verify_repair": {
            "correct": vr_correct,
            "repaired": repair_result["repaired"],
            "n_iters": repair_result["n_iters"],
            "time_s": round(repair_time, 4),
        },
    }


# ===========================================================================
# 10. Bootstrap CI (identical to Exp 162)
# ===========================================================================


def bootstrap_ci(
    correct_flags: list[bool],
    n_resamples: int = N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Compute a 95% non-parametric bootstrap confidence interval for accuracy."""
    arr = np.array(correct_flags, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)
    sample_means = np.array([arr[rng.integers(0, n, size=n)].mean() for _ in range(n_resamples)])
    alpha = 1.0 - confidence
    return (
        float(np.percentile(sample_means, 100 * alpha / 2)),
        float(np.percentile(sample_means, 100 * (1 - alpha / 2))),
    )


# ===========================================================================
# 11. NEW: Paired sign permutation test (the key fix from Exp 162)
# ===========================================================================


def paired_sign_permutation_test(
    paired_deltas: list[float],
    n_permutations: int = N_PERMUTATION,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """One-sample sign permutation test on N paired improvement deltas.

    **Detailed explanation for engineers:**
        This is the primary statistical fix over Exp 162. In Exp 162 the
        permutation test pooled only N=2 control and N=6 adversarial *aggregate*
        delta points (one pp number per model×variant), giving only C(8,2)=28
        distinct permutation states — far too few for p<0.05 resolution.

        In Exp 178 we compute one delta PER QUESTION:
            delta_q = improvement_adv_q − improvement_ctrl_q
        where improvement_q = int(vr_correct_q) − int(baseline_correct_q) ∈ {-1, 0, 1}.

        With N=400 paired deltas the sign permutation has 2^400 possible
        sign-flip combinations, which the Monte Carlo with 10,000 samples
        approximates accurately.

        Procedure (Edgington 1964, one-sample sign test):
        1. Compute observed_stat = mean(paired_deltas).
        2. For each permutation: flip each delta's sign independently with p=0.5.
        3. Compute flipped mean.
        4. p = fraction of permutations where flipped mean ≥ observed mean.

        Under H0 (no adversarial effect), flipped and original deltas should
        be equally likely, so the test is exact.

    Args:
        paired_deltas: List of N per-question improvement deltas (adv − ctrl).
        n_permutations: Number of sign-flip resamplings (default 10,000).
        seed: RNG seed for reproducibility.

    Returns:
        Dict with observed_stat, p_value, significant_p05, n, n_permutations,
        interpretation string.
    """
    rng = np.random.default_rng(seed)
    deltas = np.array(paired_deltas, dtype=float)
    n = len(deltas)
    observed_stat = float(deltas.mean())

    # Sign-flip permutation: each of N deltas independently flipped with p=0.5.
    # Shape: (n_permutations, n) binary sign matrix (0 → keep, 1 → flip).
    # Each flipped mean = mean of deltas * signs, where sign ∈ {-1, +1}.
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    null_stats = (signs * deltas[np.newaxis, :]).mean(axis=1)

    p_value = float((null_stats >= observed_stat).mean())

    if observed_stat > 0 and p_value < 0.05:
        interp = (
            f"HYPOTHESIS SUPPORTED (p={p_value:.4f} < 0.05): adversarial improvement "
            f"delta (mean={observed_stat:+.4f}) is significantly greater than control."
        )
    elif observed_stat > 0:
        interp = (
            f"Positive direction (mean delta={observed_stat:+.4f}) but NOT significant "
            f"at p<0.05 (p={p_value:.4f})."
        )
    else:
        interp = (
            f"Hypothesis NOT supported: mean paired delta = {observed_stat:+.4f} ≤ 0 "
            f"(p={p_value:.4f})."
        )

    return {
        "observed_stat": round(observed_stat, 6),
        "p_value": round(p_value, 4),
        "significant_p05": bool(p_value < 0.05),
        "n": n,
        "n_permutations": n_permutations,
        "interpretation": interp,
    }


# ===========================================================================
# 12. Two-proportion z-test (identical to Exp 162)
# ===========================================================================


def two_proportion_ztest(
    n_success_1: int,
    n_1: int,
    n_success_2: int,
    n_2: int,
    one_sided: bool = True,
) -> dict[str, Any]:
    """Two-proportion z-test: H1 p2 > p1 (one-sided by default)."""
    p1 = n_success_1 / n_1 if n_1 > 0 else 0.0
    p2 = n_success_2 / n_2 if n_2 > 0 else 0.0
    p_pool = (n_success_1 + n_success_2) / (n_1 + n_2) if (n_1 + n_2) > 0 else 0.5
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_1 + 1 / n_2)) if (n_1 > 0 and n_2 > 0) else 1.0
    z_stat = (p2 - p1) / se if se > 0 else 0.0
    p_two_sided = float(math.erfc(abs(z_stat) / math.sqrt(2)))
    p_one_sided = p_two_sided / 2.0 if z_stat > 0 else 1.0 - p_two_sided / 2.0
    p_value = p_one_sided if one_sided else p_two_sided
    return {
        "z_stat": round(z_stat, 4),
        "p_value": round(p_value, 4),
        "p1_improvement_rate": round(p1, 4),
        "p2_improvement_rate": round(p2, 4),
        "significant_p05": bool(p_value < 0.05),
        "one_sided": one_sided,
    }


# ===========================================================================
# 13. Per-variant metrics (identical to Exp 162)
# ===========================================================================


def compute_variant_metrics(
    item_results: list[dict[str, Any]],
    control_baseline_acc: float | None,
) -> dict[str, Any]:
    """Compute per-mode accuracy, CI, error breakdown for one variant."""
    n = len(item_results)
    baseline_flags = [r["baseline"]["correct"] for r in item_results]
    verify_flags = [r["verify"]["correct"] for r in item_results]
    vr_flags = [r["verify_repair"]["correct"] for r in item_results]

    def acc_metrics(flags: list[bool]) -> dict[str, Any]:
        acc = sum(flags) / n if n > 0 else 0.0
        ci_lo, ci_hi = bootstrap_ci(flags)
        drop = (control_baseline_acc - acc) * 100.0 if control_baseline_acc is not None else None
        return {
            "accuracy": round(acc, 4),
            "accuracy_pct": round(acc * 100, 2),
            "accuracy_drop_pp": round(drop, 2) if drop is not None else None,
            "ci_95_lo": round(ci_lo, 4),
            "ci_95_hi": round(ci_hi, 4),
        }

    baseline_m = acc_metrics(baseline_flags)
    verify_m = acc_metrics(verify_flags)
    vr_m = acc_metrics(vr_flags)

    baseline_acc = baseline_m["accuracy"]
    vr_acc = vr_m["accuracy"]
    delta_pp = round((vr_acc - baseline_acc) * 100.0, 2)
    verify_m["delta_pp_vs_baseline"] = round((verify_m["accuracy"] - baseline_acc) * 100.0, 2)
    vr_m["delta_pp_vs_baseline"] = delta_pp

    n_abstained = sum(r["verify"]["abstained"] for r in item_results)
    n_verified = n - n_abstained
    verify_m["n_abstained"] = n_abstained
    verify_m["abstention_rate"] = round(n_abstained / n, 4) if n > 0 else 0.0
    verified_correct = sum(
        1 for r in item_results if not r["verify"]["abstained"] and r["baseline"]["correct"]
    )
    verify_m["precision"] = round(verified_correct / n_verified, 4) if n_verified > 0 else 0.0

    error_types = ["arithmetic_error", "irrelevant_number_error", "logic_error", "reading_comprehension_error"]
    error_counts: dict[str, int] = {et: 0 for et in error_types}
    for r in item_results:
        et = r.get("error_type")
        if et and et in error_counts:
            error_counts[et] += 1

    n_errors = sum(error_counts.values())
    ising_catchable = error_counts.get("arithmetic_error", 0)
    ising_uncatchable = n_errors - ising_catchable

    irrel_total = error_counts.get("irrelevant_number_error", 0)
    irrel_passed_ising = sum(
        1 for r in item_results
        if r.get("error_type") == "irrelevant_number_error" and r["verify"]["verified"]
    )
    irrel_pass_rate = round(irrel_passed_ising / irrel_total, 4) if irrel_total > 0 else None

    n_improved_by_vr = sum(
        1 for r in item_results if r["verify_repair"]["correct"] and not r["baseline"]["correct"]
    )

    return {
        "n_questions": n,
        "baseline": baseline_m,
        "verify": verify_m,
        "verify_repair": vr_m,
        "improvement_delta_pp": delta_pp,
        "n_improved_by_vr": n_improved_by_vr,
        "error_counts": error_counts,
        "error_pcts": {et: round(c / n * 100, 1) for et, c in error_counts.items()},
        "ising_catchable_count": ising_catchable,
        "ising_uncatchable_count": ising_uncatchable,
        "ising_catchable_pct": round(ising_catchable / n * 100, 1),
        "irrelevant_pass_rate": irrel_pass_rate,
        "irrelevant_pass_n": irrel_passed_ising,
        "irrelevant_total_n": irrel_total,
    }


# ===========================================================================
# 14. Adversarial ratio (identical to Exp 162)
# ===========================================================================


def compute_adversarial_ratio(all_model_results: dict[str, Any]) -> dict[str, Any]:
    """Compute the adversarial vs. standard improvement ratio (thesis metric)."""
    per_model: dict[str, Any] = {}
    all_ratios: list[float] = []

    for model_name, mdata in all_model_results.items():
        ctrl_delta = mdata["variants"]["control"]["metrics"]["improvement_delta_pp"]
        adv_deltas = [
            mdata["variants"][vk]["metrics"]["improvement_delta_pp"]
            for vk in ADVERSARIAL_VARIANT_KEYS
        ]
        mean_adv = sum(adv_deltas) / len(adv_deltas) if adv_deltas else 0.0
        ratio = round(mean_adv / ctrl_delta, 3) if ctrl_delta > 0 else None
        per_model[model_name] = {
            "control_delta_pp": ctrl_delta,
            "mean_adversarial_delta_pp": round(mean_adv, 2),
            "adversarial_vs_standard_ratio": ratio,
            "adversarial_deltas_pp": adv_deltas,
        }
        if ratio is not None:
            all_ratios.append(ratio)

    pooled_ratio = round(sum(all_ratios) / len(all_ratios), 3) if all_ratios else None
    return {
        "per_model": per_model,
        "pooled_ratio": pooled_ratio,
        "interpretation": (
            f"Verify-repair is {pooled_ratio:.2f}× more beneficial on adversarial "
            f"inputs than on standard control inputs."
        ) if pooled_ratio is not None else "Insufficient data.",
    }


# ===========================================================================
# 15. Exp 122 replication check (identical to Exp 162)
# ===========================================================================


def check_exp122_replication(all_model_results: dict[str, Any]) -> dict[str, Any]:
    """Check whether the Exp 122 finding (74% irrelevant-error pass-through) replicates."""
    exp122_reference = 0.74
    total_irrel = 0
    total_passed = 0

    for _model_name, mdata in all_model_results.items():
        for vk in ("irrelevant_injected", "combined"):
            vm = mdata["variants"][vk]["metrics"]
            total_passed += vm.get("irrelevant_pass_n", 0)
            total_irrel += vm.get("irrelevant_total_n", 0)

    observed = round(total_passed / total_irrel, 4) if total_irrel > 0 else None
    matches = abs(observed - exp122_reference) < 0.10 if observed is not None else False
    return {
        "exp122_reference_rate": exp122_reference,
        "observed_pass_rate": observed,
        "total_irrelevant_errors": total_irrel,
        "total_passed_ising": total_passed,
        "matches_exp122": matches,
        "interpretation": (
            f"Observed {observed:.1%} pass-through rate for irrelevant-number errors "
            f"vs. Exp 122 reference of {exp122_reference:.1%}. "
            + ("Replicates." if matches else "DEVIATION — investigate simulation.")
        ) if observed is not None else "No irrelevant-number errors observed.",
    }


# ===========================================================================
# 16. NEW: Per-question paired delta extraction
# ===========================================================================


def extract_paired_deltas(
    ctrl_item_results: list[dict[str, Any]],
    adv_item_results: list[dict[str, Any]],
) -> list[float]:
    """Compute per-question improvement deltas: adv_improvement_q − ctrl_improvement_q.

    **Detailed explanation for engineers:**
        improvement_q = int(vr_correct_q) − int(baseline_correct_q) ∈ {-1, 0, 1}
            +1: baseline wrong → VR correct  (VR helped)
             0: both wrong or both correct   (VR neutral)
            -1: baseline correct → VR wrong  (VR hurt, rare)

        paired_delta_q = adv_improvement_q − ctrl_improvement_q ∈ {-2, -1, 0, 1, 2}
            positive: VR helps MORE on this adversarial question than on the control
            zero:     VR helps equally (or neither helped on both)
            negative: VR helps MORE on the control question than on the adversarial

        H1: mean(paired_delta) > 0 (VR systematically more beneficial on adversarial).

        Both lists must be the same length N (enforced by design; both use items
        from positions 0..N-1 in their respective datasets, which share the same
        underlying template structure).

    Returns:
        List of N float paired deltas, one per question.
    """
    assert len(ctrl_item_results) == len(adv_item_results), (
        f"ctrl length {len(ctrl_item_results)} ≠ adv length {len(adv_item_results)}"
    )
    deltas: list[float] = []
    for ctrl_r, adv_r in zip(ctrl_item_results, adv_item_results):
        ctrl_improvement = int(ctrl_r["verify_repair"]["correct"]) - int(ctrl_r["baseline"]["correct"])
        adv_improvement = int(adv_r["verify_repair"]["correct"]) - int(adv_r["baseline"]["correct"])
        deltas.append(float(adv_improvement - ctrl_improvement))
    return deltas


# ===========================================================================
# 17. Main experiment loop
# ===========================================================================


def run_experiment() -> dict[str, Any]:
    """Run Experiment 178: Definitive Adversarial GSM8K — N=400, Paired Permutation.

    Design:
        Outer loop: 2 models.
        Inner loop: 4 variants × 400 questions = 1,600 per model.
        After all models:
            Per adversarial variant:
                - Paired sign permutation test (N=400 paired deltas, 10,000 resamplings)
                - Two-proportion z-test (per-question improvement flags)
            Goal #5 achieved: BOTH tests p<0.05 for ≥1 adversarial variant.
    """
    t_exp_start = time.time()
    print("=" * 76)
    print("Experiment 178: Definitive Adversarial GSM8K (N=400, Paired Permutation)")
    print("=" * 76)
    print()
    print("H1: verify-repair Δ per question is LARGER on adversarial than on control.")
    print("    Primary test: paired sign permutation (N=400, 10k resamplings).")
    print("    Secondary test: two-proportion z-test (convergent validity).")
    print("    Goal #5 requires BOTH tests p<0.05 for ≥1 adversarial variant.")
    print()

    print("[1] Loading and augmenting adversarial datasets...")
    datasets = load_and_augment_data()
    n_per_variant = len(datasets["control"])
    print(f"  Final N: {n_per_variant} items/variant ({n_per_variant * 4} total per model)")
    print()

    all_model_results: dict[str, Any] = {}
    # Store per-question item results for paired test (indexed by [model][variant]).
    all_item_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    inference_modes: list[str] = []

    # For legacy z-test (pooled across all adversarial variants and both models).
    ctrl_improved_total = 0
    ctrl_n_total = 0
    adv_improved_total = 0
    adv_n_total = 0

    for model_cfg in MODEL_CONFIGS:
        model_name = model_cfg["name"]
        base_error_rate = model_cfg["base_error_rate"]

        print(f"[Model: {model_name}]")
        tokenizer, model_obj, device, use_live = load_model(model_cfg)
        inference_mode = "live" if use_live else "simulated"
        inference_modes.append(inference_mode)
        if use_live:
            print(f"  Live model loaded on {device}.")
        else:
            print(f"  Using adversarial simulation (Apple-calibrated error rates).")

        # Seeds: same convention as Exp 162 + offset for augmented items.
        # sim_rng advances through all 400 items; RNG state is continuous.
        sim_seed = sum(ord(c) for c in model_name) + 178
        repair_seed = sum(ord(c) for c in model_name) + 178 + 1000
        sim_rng = random.Random(sim_seed)
        repair_rng = random.Random(repair_seed)

        model_variant_results: dict[str, Any] = {}
        variant_item_results: dict[str, list[dict[str, Any]]] = {}
        control_baseline_acc: float | None = None

        for variant_key, variant_label, multiplier in VARIANTS:
            items = datasets[variant_key]
            n = len(items)
            print(f"\n  Variant: {variant_label} ({n} questions, ×{multiplier:.1f} err mult)")

            item_results: list[dict[str, Any]] = []
            t_var = time.time()

            for i, item in enumerate(items):
                if (i + 1) % 100 == 0:
                    print(f"    [{i+1}/{n}] {time.time()-t_var:.1f}s elapsed...")
                result = evaluate_item(
                    item, variant_key, multiplier,
                    model_name, base_error_rate,
                    tokenizer, model_obj, device, use_live,
                    sim_rng, repair_rng,
                )
                item_results.append(result)

            t_var_elapsed = time.time() - t_var
            metrics = compute_variant_metrics(item_results, control_baseline_acc)

            if variant_key == "control":
                control_baseline_acc = metrics["baseline"]["accuracy"]

            b = metrics["baseline"]
            vr = metrics["verify_repair"]
            delta = metrics["improvement_delta_pp"]
            print(
                f"    Baseline: {b['accuracy_pct']:.1f}%"
                + (f" (drop: {b['accuracy_drop_pp']:+.1f}pp vs ctrl)" if b["accuracy_drop_pp"] is not None else "")
                + f"  →  VR: {vr['accuracy_pct']:.1f}%"
                + f"  Δ={delta:+.1f}pp  [{t_var_elapsed:.1f}s]"
            )
            ec = metrics["error_counts"]
            irr = metrics.get("irrelevant_pass_rate")
            irr_str = f", irrel_pass={irr:.0%}" if irr is not None else ""
            print(
                f"    Errors — arith: {ec['arithmetic_error']} (Ising↑)"
                f", irrel: {ec['irrelevant_number_error']} (Ising∅{irr_str})"
                f", logic: {ec['logic_error']}, read: {ec['reading_comprehension_error']}"
            )

            variant_item_results[variant_key] = item_results
            model_variant_results[variant_key] = {
                "variant_label": variant_label,
                "multiplier": multiplier,
                "metrics": metrics,
                "inference_mode": inference_mode,
            }

            # Accumulate z-test counters.
            n_improved = metrics["n_improved_by_vr"]
            if variant_key == "control":
                ctrl_improved_total += n_improved
                ctrl_n_total += n
            else:
                adv_improved_total += n_improved
                adv_n_total += n

        if use_live:
            unload_model(model_obj, tokenizer, device)

        all_item_results[model_name] = variant_item_results
        ctrl_delta = model_variant_results["control"]["metrics"]["improvement_delta_pp"]
        adv_deltas_pp = [
            model_variant_results[vk]["metrics"]["improvement_delta_pp"]
            for vk in ADVERSARIAL_VARIANT_KEYS
        ]
        all_model_results[model_name] = {
            "model_config": {
                "name": model_name,
                "candidates": model_cfg["candidates"],
                "base_error_rate": base_error_rate,
            },
            "inference_mode": inference_mode,
            "variants": model_variant_results,
            "control_delta_pp": ctrl_delta,
            "adversarial_deltas_pp": adv_deltas_pp,
        }
        print()

    # -----------------------------------------------------------------------
    # Hypothesis tests: paired sign permutation + z-test per adversarial variant
    # -----------------------------------------------------------------------
    print("[4] Running hypothesis tests (paired permutation + z-test per variant)...")

    # Aggregate ctrl item_results across both models (for pooled paired test).
    # Per-variant tests use all available paired data pooled across models.
    per_variant_paired_tests: dict[str, dict[str, Any]] = {}
    per_variant_ztest: dict[str, dict[str, Any]] = {}

    for adv_vk in ADVERSARIAL_VARIANT_KEYS:
        # Pool paired deltas across both models.
        all_paired_deltas: list[float] = []
        for model_name in all_model_results:
            ctrl_results = all_item_results[model_name]["control"]
            adv_results  = all_item_results[model_name][adv_vk]
            paired_deltas = extract_paired_deltas(ctrl_results, adv_results)
            all_paired_deltas.extend(paired_deltas)

        perm_result = paired_sign_permutation_test(
            all_paired_deltas, n_permutations=N_PERMUTATION, seed=PERMUTATION_SEED,
        )
        per_variant_paired_tests[adv_vk] = perm_result

        # Per-variant z-test: compare improvement rates adv vs ctrl.
        ctrl_improved_var = 0
        ctrl_n_var = 0
        adv_improved_var = 0
        adv_n_var = 0
        for model_name in all_model_results:
            ctrl_met = all_model_results[model_name]["variants"]["control"]["metrics"]
            adv_met  = all_model_results[model_name]["variants"][adv_vk]["metrics"]
            ctrl_improved_var += ctrl_met["n_improved_by_vr"]
            ctrl_n_var        += ctrl_met["n_questions"]
            adv_improved_var  += adv_met["n_improved_by_vr"]
            adv_n_var         += adv_met["n_questions"]

        ztest_result = two_proportion_ztest(
            n_success_1=ctrl_improved_var,
            n_1=ctrl_n_var,
            n_success_2=adv_improved_var,
            n_2=adv_n_var,
            one_sided=True,
        )
        per_variant_ztest[adv_vk] = ztest_result

        sig_perm = perm_result["significant_p05"]
        sig_z    = ztest_result["significant_p05"]
        marker = "✓ BOTH" if (sig_perm and sig_z) else ("perm✓" if sig_perm else ("z✓" if sig_z else "✗ neither"))
        print(
            f"  {adv_vk:22s} | perm p={perm_result['p_value']:.4f}"
            f" ({'' if sig_perm else 'NOT '}sig)"
            f" | z p={ztest_result['p_value']:.4f}"
            f" ({'' if sig_z else 'NOT '}sig)"
            f" | {marker}"
        )

    # Pooled legacy z-test (for comparison with Exp 162).
    pooled_ztest = two_proportion_ztest(
        n_success_1=ctrl_improved_total, n_1=ctrl_n_total,
        n_success_2=adv_improved_total, n_2=adv_n_total,
        one_sided=True,
    )

    # Goal #5: achieved if BOTH tests p<0.05 for at least one adversarial variant.
    goal5_achieved = any(
        per_variant_paired_tests[vk]["significant_p05"] and per_variant_ztest[vk]["significant_p05"]
        for vk in ADVERSARIAL_VARIANT_KEYS
    )

    # Best variant for Goal #5 reporting.
    best_variant_key = min(
        ADVERSARIAL_VARIANT_KEYS,
        key=lambda vk: max(
            per_variant_paired_tests[vk]["p_value"],
            per_variant_ztest[vk]["p_value"],
        ),
    )

    adversarial_ratio = compute_adversarial_ratio(all_model_results)
    exp122_check = check_exp122_replication(all_model_results)

    print()
    print(f"  Adv/ctrl ratio:  {adversarial_ratio['pooled_ratio']:.2f}×")
    print(f"  Exp 122 check:   {exp122_check['interpretation']}")
    print(f"  *** GOAL #5 ACHIEVED: {goal5_achieved} ***")
    print()

    # -----------------------------------------------------------------------
    # Summary tables
    # -----------------------------------------------------------------------
    per_variant_accuracy: dict[str, dict[str, Any]] = {}
    for vk, vlabel, _ in VARIANTS:
        per_variant_accuracy[vk] = {"label": vlabel}
        for model_name in all_model_results:
            m = all_model_results[model_name]["variants"][vk]["metrics"]
            per_variant_accuracy[vk][model_name] = {
                "baseline_pct": m["baseline"]["accuracy_pct"],
                "verify_repair_pct": m["verify_repair"]["accuracy_pct"],
                "delta_pp": m["improvement_delta_pp"],
                "ci_95": [m["verify_repair"]["ci_95_lo"] * 100, m["verify_repair"]["ci_95_hi"] * 100],
            }

    improvement_deltas: dict[str, Any] = {}
    for model_name in all_model_results:
        improvement_deltas[model_name] = {
            vk: all_model_results[model_name]["variants"][vk]["metrics"]["improvement_delta_pp"]
            for vk, _, _ in VARIANTS
        }

    error_taxonomy: dict[str, Any] = {}
    for model_name in all_model_results:
        error_taxonomy[model_name] = {}
        for vk, _, _ in VARIANTS:
            m = all_model_results[model_name]["variants"][vk]["metrics"]
            error_taxonomy[model_name][vk] = {
                "arithmetic_error": m["error_counts"]["arithmetic_error"],
                "irrelevant_number_error": m["error_counts"]["irrelevant_number_error"],
                "logic_error": m["error_counts"]["logic_error"],
                "reading_comprehension_error": m["error_counts"]["reading_comprehension_error"],
                "ising_catchable_pct": m["ising_catchable_pct"],
                "irrelevant_pass_rate": m.get("irrelevant_pass_rate"),
            }

    t_exp_elapsed = time.time() - t_exp_start

    # Per-variant summary for the output spec.
    per_variant_summary: dict[str, dict[str, Any]] = {}
    for vk in ADVERSARIAL_VARIANT_KEYS:
        # Pool across models for aggregate values.
        ctrl_baselines = [all_model_results[m]["variants"]["control"]["metrics"]["baseline"]["accuracy_pct"]
                          for m in all_model_results]
        adv_baselines  = [all_model_results[m]["variants"][vk]["metrics"]["baseline"]["accuracy_pct"]
                          for m in all_model_results]
        ctrl_repaired  = [all_model_results[m]["variants"]["control"]["metrics"]["verify_repair"]["accuracy_pct"]
                          for m in all_model_results]
        adv_repaired   = [all_model_results[m]["variants"][vk]["metrics"]["verify_repair"]["accuracy_pct"]
                          for m in all_model_results]
        ctrl_delta_pp_list = [all_model_results[m]["variants"]["control"]["metrics"]["improvement_delta_pp"]
                              for m in all_model_results]
        adv_delta_pp_list  = [all_model_results[m]["variants"][vk]["metrics"]["improvement_delta_pp"]
                              for m in all_model_results]

        mean_ctrl_delta = sum(ctrl_delta_pp_list) / len(ctrl_delta_pp_list)
        mean_adv_delta  = sum(adv_delta_pp_list)  / len(adv_delta_pp_list)
        # CI from Exp 162-style bootstrap (use combined VR flags for CI).
        ci_95_list = [
            [all_model_results[m]["variants"][vk]["metrics"]["verify_repair"]["ci_95_lo"] * 100,
             all_model_results[m]["variants"][vk]["metrics"]["verify_repair"]["ci_95_hi"] * 100]
            for m in all_model_results
        ]
        per_variant_summary[vk] = {
            "baseline_accuracy_pct_per_model": dict(zip(all_model_results.keys(), adv_baselines)),
            "repair_accuracy_pct_per_model":   dict(zip(all_model_results.keys(), adv_repaired)),
            "improvement_delta_pp_per_model":  dict(zip(all_model_results.keys(), adv_delta_pp_list)),
            "mean_improvement_delta_pp": round(mean_adv_delta, 2),
            "mean_ctrl_delta_pp": round(mean_ctrl_delta, 2),
            "ci_95_per_model": dict(zip(all_model_results.keys(), ci_95_list)),
            "p_permutation": per_variant_paired_tests[vk]["p_value"],
            "p_ztest":        per_variant_ztest[vk]["p_value"],
            "significant_permutation": per_variant_paired_tests[vk]["significant_p05"],
            "significant_ztest":       per_variant_ztest[vk]["significant_p05"],
            "goal5_this_variant": bool(
                per_variant_paired_tests[vk]["significant_p05"]
                and per_variant_ztest[vk]["significant_p05"]
            ),
        }

    inference_mode_final = "live" if any(m == "live" for m in inference_modes) else "simulated"

    return {
        "experiment": 178,
        "description": (
            "Definitive adversarial GSM8K benchmark: N=400/variant, paired sign "
            "permutation test (N=400 per-question deltas), two-proportion z-test. "
            "Fixes Exp 162's underpowered permutation test (N=2 ctrl + 6 adv points). "
            "Goal #5 requires BOTH tests p<0.05 for ≥1 adversarial variant."
        ),
        "reference": (
            "Apple arxiv 2410.05229; Exp 162 (N=200, flawed permutation test); "
            "Exp 119 (dataset generation); Exp 177 (GPU confirmation)"
        ),
        "hypothesis": (
            "H1: per-question verify-repair improvement delta is LARGER on adversarial "
            "variants than on control. N=400 paired sign permutation test, powered for "
            "p<0.05 at the Exp 162 observed effect size."
        ),
        # Required output fields (per task spec).
        "n_per_variant": n_per_variant,
        "inference_mode": inference_mode_final,
        "per_variant": per_variant_summary,
        "adversarial_control_ratio": adversarial_ratio["pooled_ratio"],
        "goal5_achieved": goal5_achieved,
        # Detailed model-level results.
        "models": all_model_results,
        "variant_keys": [v[0] for v in VARIANTS],
        "modes": [m[0] for m in MODES],
        "per_variant_accuracy": per_variant_accuracy,
        "improvement_deltas": improvement_deltas,
        # Per-variant statistical tests (primary output).
        "per_variant_paired_tests": per_variant_paired_tests,
        "per_variant_ztest": per_variant_ztest,
        "pooled_ztest": pooled_ztest,
        "best_variant_for_goal5": best_variant_key,
        "ci_95": {
            model_name: {
                vk: [
                    all_model_results[model_name]["variants"][vk]["metrics"]["verify_repair"]["ci_95_lo"],
                    all_model_results[model_name]["variants"][vk]["metrics"]["verify_repair"]["ci_95_hi"],
                ]
                for vk, _, _ in VARIANTS
            }
            for model_name in all_model_results
        },
        "error_taxonomy": error_taxonomy,
        "adversarial_vs_standard_ratio": adversarial_ratio,
        "exp122_replication": exp122_check,
        # Exp 162 comparison.
        "exp162_permutation_p": 0.4289,
        "exp162_ztest_p": 0.0174,
        "exp162_n_per_variant": 200,
        "design_improvement": (
            "Paired per-question permutation test replaces aggregate per-variant "
            "delta test. N=400 paired deltas vs Exp 162's N=8 aggregate points."
        ),
        "n_permutations": N_PERMUTATION,
        "experiment_elapsed_s": round(t_exp_elapsed, 1),
    }


# ===========================================================================
# 18. Results table printer
# ===========================================================================


def print_results_table(results: dict[str, Any]) -> None:
    """Print comprehensive human-readable results tables."""
    variant_keys = results["variant_keys"]
    models = list(results["models"].keys())

    SEP = "=" * 84

    print(SEP)
    print("EXP 178 — ACCURACY BY MODE (N=400/variant)")
    print(SEP)
    print(f"{'Model/Mode':<26} {'Control':>12} {'Num-Swap':>12} {'Irrel-Inj':>12} {'Combined':>12}")
    print("-" * 84)
    for model_name in models:
        mdata = results["models"][model_name]
        for mode_key, mode_label in [("baseline", "Baseline"), ("verify_repair", "→ VR")]:
            row = f"  {(model_name[:18] + ' ' + mode_label[:5]):<24}"
            for vk in variant_keys:
                acc = mdata["variants"][vk]["metrics"][mode_key]["accuracy_pct"]
                row += f" {acc:>11.1f}%"
            print(row)
        row_ci = f"    {'(95% CI)':>22}"
        for vk in variant_keys:
            m = mdata["variants"][vk]["metrics"]["verify_repair"]
            row_ci += f"  [{m['ci_95_lo']*100:.0f}–{m['ci_95_hi']*100:.0f}%]"
        print(row_ci)
        print()

    print(SEP)
    print("IMPROVEMENT DELTA (Verify-Repair − Baseline, pp)")
    print(SEP)
    print(f"{'Model':<26} {'Control':>12} {'Num-Swap':>12} {'Irrel-Inj':>12} {'Combined':>12}")
    print("-" * 84)
    for model_name in models:
        mdata = results["models"][model_name]
        ctrl_delta = mdata["variants"]["control"]["metrics"]["improvement_delta_pp"]
        row = f"  {model_name:<24}"
        for vk in variant_keys:
            delta = mdata["variants"][vk]["metrics"]["improvement_delta_pp"]
            marker = "*" if vk != "control" and delta > ctrl_delta else " "
            row += f" {delta:>+10.1f}pp{marker}"
        print(row)
    print()

    print(SEP)
    print("PAIRED SIGN PERMUTATION TEST (per adversarial variant, N=400 paired deltas)")
    print("  H1: mean(adv_improvement_q − ctrl_improvement_q) > 0 (one-sided)")
    print(SEP)
    print(f"{'Variant':<24} {'N deltas':>10} {'Obs stat':>12} {'p-value':>10} {'sig?':>8}")
    print("-" * 84)
    for adv_vk in ADVERSARIAL_VARIANT_KEYS:
        pt = results["per_variant_paired_tests"][adv_vk]
        sig = "YES ***" if pt["significant_p05"] else "no"
        print(
            f"  {adv_vk:<22} {pt['n']:>10,} {pt['observed_stat']:>+12.5f}"
            f" {pt['p_value']:>10.4f} {sig:>8}"
        )
    print()

    print(SEP)
    print("TWO-PROPORTION Z-TEST (per adversarial variant)")
    print("  H1: improvement rate on adversarial > improvement rate on control (one-sided)")
    print(SEP)
    print(f"{'Variant':<24} {'ctrl rate':>12} {'adv rate':>12} {'z-stat':>10} {'p-value':>10} {'sig?':>8}")
    print("-" * 84)
    for adv_vk in ADVERSARIAL_VARIANT_KEYS:
        zt = results["per_variant_ztest"][adv_vk]
        sig = "YES ***" if zt["significant_p05"] else "no"
        print(
            f"  {adv_vk:<22}"
            f" {zt['p1_improvement_rate']:>12.4f}"
            f" {zt['p2_improvement_rate']:>12.4f}"
            f" {zt['z_stat']:>10.4f}"
            f" {zt['p_value']:>10.4f}"
            f" {sig:>8}"
        )
    print()

    print(SEP)
    print("GOAL #5 STATUS (requires BOTH tests p<0.05 for ≥1 adversarial variant)")
    print(SEP)
    print(f"{'Variant':<24} {'perm p':>10} {'z p':>10} {'perm sig':>10} {'z sig':>8} {'BOTH?':>8}")
    print("-" * 84)
    for adv_vk in ADVERSARIAL_VARIANT_KEYS:
        pt = results["per_variant_paired_tests"][adv_vk]
        zt = results["per_variant_ztest"][adv_vk]
        both = pt["significant_p05"] and zt["significant_p05"]
        print(
            f"  {adv_vk:<22}"
            f" {pt['p_value']:>10.4f}"
            f" {zt['p_value']:>10.4f}"
            f" {'YES' if pt['significant_p05'] else 'no':>10}"
            f" {'YES' if zt['significant_p05'] else 'no':>8}"
            f" {'*** YES' if both else 'no':>8}"
        )
    print()
    print(f"  Adversarial/control improvement ratio: {results['adversarial_control_ratio']:.2f}×")
    print(f"  Exp 122 replication: {results['exp122_replication']['interpretation']}")
    print()
    print(f"  {'★ GOAL #5 ACHIEVED' if results['goal5_achieved'] else '✗ Goal #5 NOT YET ACHIEVED'}")
    print(f"  Experiment elapsed: {results['experiment_elapsed_s']:.1f}s")
    print(SEP)


# ===========================================================================
# 19. Entry point
# ===========================================================================


def main() -> None:
    """Run experiment, print results, save JSON."""
    print(f"\nResults will be saved to: {OUTPUT_PATH}\n")

    results = run_experiment()
    print_results_table(results)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
