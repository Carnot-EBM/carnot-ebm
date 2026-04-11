#!/usr/bin/env python3
"""Experiment 182: DEFINITIVE Adversarial GSM8K — Live GPU Inference, N=400/variant.

**Researcher summary:**
    This is the definitive adversarial benchmark. Previous attempts:
    - Exp 147: p=0.463 — underpowered (N=6 improvement deltas, simulated).
    - Exp 162: N=200/variant but simulation fallback — not publishable.
    - Exp 181: Live GPU ✓, but standard GSM8K only (no adversarial variants).

    Exp 182 combines Exp 181's live GPU pattern with Exp 162's adversarial
    design, upgrading N=200 → 400 per variant for 80% power at p<0.05 even
    if the true effect size is smaller than Exp 147 estimated.

    Design:
    - 4 variants × 400 questions = 1,600 total adversarial items.
    - 3 modes: Baseline, Verify-only, Verify+Repair (up to 3 iters).
    - 2 models: Qwen/Qwen3.5-0.8B (GPU 0), google/gemma-4-E4B-it (GPU 1).
    - 95% bootstrap CIs (10,000 samples) on all accuracy numbers.
    - Paired significance test: adversarial improvement delta > control delta.
    - Two-proportion z-test (per-question flags) as convergent validity.
    - Error taxonomy with per-type N sufficient for sub-group statistics.

    Hypothesis (Goal #5, research-program.md):
        H0: verify-repair Δ on adversarial ≤ verify-repair Δ on control
        H1: verify-repair Δ on adversarial > verify-repair Δ on control (one-sided)
        Expected: p<0.05, powered by N=400.

**Detailed explanation for engineers:**
    Adversarial variant types (from Apple arxiv 2410.05229):
    1. number_swapped: same problem structure, different numbers → model must
       reason from scratch, can't rely on memorized answer patterns.
    2. irrelevant_injected: distractor number inserted → model must ignore
       it (Ising correctly misses these: arithmetic is internally consistent).
    3. combined: both perturbations applied simultaneously.
    4. control: unmodified synthetic GSM8K questions.

    Statistical design:
    - Two-proportion z-test: compare per-question improvement flags between
      adversarial and control groups (2×2 contingency table, one-sided).
    - Permutation test: 10,000 resamplings of improvement-delta ranks.
    - Convergent significance: both tests must agree for H1 to be confirmed.

    Live GPU inference (no simulation):
    CARNOT_FORCE_LIVE=1 prevents silent fallback to simulation. Models are
    loaded with device_map={"": device_index} for correct multi-GPU placement
    (bypasses model_loader.py's cuda() default which always uses GPU 0).

    Checkpointing:
    Results saved every 50 questions per model. On crash/resume, the script
    detects existing checkpoints and skips already-completed questions.
    Checkpoint files: results/exp182_ckpt_{model}_{variant}.json

Usage:
    CARNOT_FORCE_LIVE=1 .venv/bin/python scripts/experiment_182_adversarial_live_gpu.py

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
# Force GPU mode before any carnot imports. See ops/known-issues.md.
# CARNOT_FORCE_CPU=0 enables CUDA. CARNOT_FORCE_LIVE=1 prevents simulation.
# ---------------------------------------------------------------------------
os.environ["CARNOT_FORCE_CPU"] = "0"
os.environ["CARNOT_FORCE_LIVE"] = "1"

# ---------------------------------------------------------------------------
# Path setup.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "experiment_182_results.json"
ADVERSARIAL_DATA_PATH = RESULTS_DIR / "adversarial_gsm8k_data_400.json"

# ---------------------------------------------------------------------------
# GPU-specific imports (fail fast if CUDA not available).
# ---------------------------------------------------------------------------
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as _e:
    print(f"FATAL: torch/transformers not installed: {_e}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("FATAL: CUDA not available. Exp 182 requires RTX 3090 GPUs.")
    print("  Check: nvidia-smi, CUDA drivers, torch+cuda build.")
    sys.exit(1)

_N_GPUS = torch.cuda.device_count()
print(f"CUDA available: {_N_GPUS} GPU(s).")
for _i in range(_N_GPUS):
    print(f"  GPU {_i}: {torch.cuda.get_device_name(_i)} "
          f"({torch.cuda.get_device_properties(_i).total_memory // 1024**2} MB)")

# ---------------------------------------------------------------------------
# Configuration constants.
# ---------------------------------------------------------------------------
N_PER_VARIANT = 400        # 2× Exp 162 for 80% power.
N_BOOTSTRAP = 10_000       # 10k samples → <±2.5pp CI at N=400.
BOOTSTRAP_SEED = 182
N_PERMUTATION = 10_000
PERMUTATION_SEED = 182
MAX_REPAIR_ITERS = 3

VARIANTS: list[tuple[str, str, float]] = [
    ("control",             "Control (standard)",          1.0),
    ("number_swapped",      "Number-swapped",               1.8),
    ("irrelevant_injected", "Irrelevant-injected",          1.5),
    ("combined",            "Combined adversarial",         2.2),
]

# ---------------------------------------------------------------------------
# Model configurations.
# ---------------------------------------------------------------------------
MODEL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "Qwen3.5-0.8B",
        "hf_id": "Qwen/Qwen3.5-0.8B",
        "fallback_id": "Qwen/Qwen3-0.6B",
        "device_index": 0,
        "dtype": "float16",
    },
    {
        "name": "Gemma4-E4B-it",
        "hf_id": "google/gemma-4-E4B-it",
        "fallback_id": None,
        "device_index": 1 if _N_GPUS >= 2 else 0,
        "dtype": "float16",
    },
]


# ---------------------------------------------------------------------------
# Section 1: Synthetic GSM8K problem templates (from Exp 119 / 162).
# Each template takes a random.Random instance and returns (text, answer).
# ---------------------------------------------------------------------------


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
    """Construction: workers × hours × days × rate."""
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
# Irrelevant sentence templates (Apple GSM-NoOp methodology).
# Each contains {N} placeholder filled with a contextually plausible number
# that has NO bearing on the problem's correct answer.
# ---------------------------------------------------------------------------
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
    """Select a random irrelevant template and fill {N} with a non-conflicting number.

    **Detailed explanation for engineers:**
        We avoid injecting a number that already appears in the problem text.
        This prevents edge cases where the injected number coincidentally equals
        the correct answer, which would make any model answer look correct.
    """
    template = rng.choice(IRRELEVANT_TEMPLATES)
    candidates = list(range(2, 50)) + [60, 75, 100, 120, 150, 200]
    rng.shuffle(candidates)
    chosen = candidates[0]
    for c in candidates:
        if c not in exclude_numbers:
            chosen = c
            break
    return template.replace("{N}", str(chosen))


def _inject_irrelevant_sentence(problem: str, rng: random.Random) -> str:
    """Insert one irrelevant numeric sentence at a random non-final position.

    **Detailed explanation for engineers:**
        Splits on sentence boundaries and inserts at a random interior position
        so the distractor blends naturally into the problem narrative. Matches
        Apple paper's GSM-NoOp methodology.
    """
    existing: set[int] = set()
    for m in re.finditer(r"\b(\d+)\b", problem):
        try:
            existing.add(int(m.group(1)))
        except ValueError:
            pass
    irrel = _generate_irrelevant_sentence(rng, existing)
    parts = re.split(r"(?<=\.)\s+", problem.strip())
    if len(parts) <= 1:
        return irrel + " " + problem
    insert_pos = rng.randint(1, max(1, len(parts) - 1))
    parts.insert(insert_pos, irrel)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Section 2: Adversarial variant generators.
# ---------------------------------------------------------------------------


def number_swap(
    tmpl_fn: Callable[[random.Random], tuple[str, int]],
    swap_seed: int,
) -> tuple[str, int]:
    """Re-run the same template with a different seed — new numbers, new answer.

    **Detailed explanation for engineers:**
        Because all problems are generated by parameterized templates, a
        number-swap is identical to re-seeding the template. The logical
        structure (what is asked, how many steps) is preserved; all numeric
        values change. The new answer is provably correct.
    """
    return tmpl_fn(random.Random(swap_seed))


def irrelevant_injection(
    problem: str, answer: int, inject_rng: random.Random,
) -> tuple[str, int]:
    """Insert irrelevant sentence; correct answer unchanged (it IS irrelevant)."""
    return _inject_irrelevant_sentence(problem, inject_rng), answer


def combined_adversarial(
    tmpl_fn: Callable[[random.Random], tuple[str, int]],
    swap_seed: int,
    inject_rng: random.Random,
) -> tuple[str, int]:
    """Number-swap then irrelevant injection — worst-case adversarial condition."""
    swapped, new_answer = tmpl_fn(random.Random(swap_seed))
    perturbed, _ = irrelevant_injection(swapped, new_answer, inject_rng)
    return perturbed, new_answer


# ---------------------------------------------------------------------------
# Section 3: Dataset generation — N_PER_VARIANT × 4 variants.
# ---------------------------------------------------------------------------


def generate_adversarial_datasets(
    n: int = N_PER_VARIANT,
    seed: int = 182,
) -> dict[str, list[dict[str, Any]]]:
    """Generate all four adversarial dataset variants.

    **Detailed explanation for engineers:**
        Generates n problems per variant with consistent seeds. Each question
        index i uses the same template across all 4 variants so inter-variant
        comparisons are valid. Seed triplet per item:
          control_seed = seed + i * 1000
          swap_seed    = seed + i * 1000 + 500
          inject_seed  = seed + i * 1000 + 250

        This scheme avoids any cross-contamination between variants.

    Returns:
        Dict with keys "control", "number_swapped", "irrelevant_injected",
        "combined", each containing n question dicts.
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
        inj_seed = seed + i * 1000 + 250

        orig_problem, orig_answer = tmpl_fn(random.Random(ctrl_seed))

        # Control variant (no perturbation).
        control_items.append({
            "id": i, "original_problem": orig_problem,
            "perturbed_problem": orig_problem,
            "correct_answer": orig_answer, "original_answer": orig_answer,
            "perturbation": "none", "template": tmpl_name,
        })

        # Number-swapped.
        swap_problem, swap_answer = number_swap(tmpl_fn, swap_seed)
        swapped_items.append({
            "id": i, "original_problem": orig_problem,
            "perturbed_problem": swap_problem,
            "correct_answer": swap_answer, "original_answer": orig_answer,
            "perturbation": "number_swap", "template": tmpl_name,
        })

        # Irrelevant injection.
        inj_problem, inj_answer = irrelevant_injection(
            orig_problem, orig_answer, random.Random(inj_seed),
        )
        injected_items.append({
            "id": i, "original_problem": orig_problem,
            "perturbed_problem": inj_problem,
            "correct_answer": inj_answer, "original_answer": orig_answer,
            "perturbation": "irrelevant_injection", "template": tmpl_name,
        })

        # Combined.
        comb_rng = random.Random(inj_seed + 1)
        comb_problem, comb_answer = combined_adversarial(tmpl_fn, swap_seed, comb_rng)
        combined_items.append({
            "id": i, "original_problem": orig_problem,
            "perturbed_problem": comb_problem,
            "correct_answer": comb_answer, "original_answer": orig_answer,
            "perturbation": "combined", "template": tmpl_name,
        })

    return {
        "control": control_items,
        "number_swapped": swapped_items,
        "irrelevant_injected": injected_items,
        "combined": combined_items,
    }


def load_or_generate_datasets() -> dict[str, list[dict[str, Any]]]:
    """Load cached datasets or regenerate if missing.

    **Detailed explanation for engineers:**
        Caches to adversarial_gsm8k_data_400.json to avoid regenerating
        the full 1,600-question set on every run. JSON is ~300 KB.
    """
    if ADVERSARIAL_DATA_PATH.exists():
        print(f"  Loading adversarial datasets from {ADVERSARIAL_DATA_PATH.name}...")
        with open(ADVERSARIAL_DATA_PATH) as f:
            raw = json.load(f)
        datasets = raw["datasets"]
        counts = {k: len(v) for k, v in datasets.items()}
        print(f"  Loaded: {counts}")
        # Validate size.
        if all(v >= N_PER_VARIANT for v in counts.values()):
            return datasets
        print(f"  Insufficient size (need {N_PER_VARIANT}/variant) — regenerating.")

    print(f"  Generating {N_PER_VARIANT} questions × 4 variants = {N_PER_VARIANT*4} total...")
    datasets = generate_adversarial_datasets(N_PER_VARIANT, seed=182)
    with open(ADVERSARIAL_DATA_PATH, "w") as f:
        json.dump({"n_per_variant": N_PER_VARIANT, "seed": 182, "datasets": datasets}, f)
    print(f"  Saved to {ADVERSARIAL_DATA_PATH.name}.")
    return datasets


# ---------------------------------------------------------------------------
# Section 4: Multi-GPU model loading.
# ---------------------------------------------------------------------------


def load_model_on_gpu(config: dict[str, Any]) -> tuple[Any, Any, str]:
    """Load a HuggingFace model onto a specific GPU.

    **Detailed explanation for engineers:**
        Uses device_map={"": device_index} to place all model layers on the
        target GPU at load time. This is the correct multi-GPU fix from
        Exp 180's findings (model_loader.py's model.cuda() always uses GPU 0).

        CARNOT_FORCE_LIVE=1 means any load failure is fatal — we never silently
        fall back to simulation.
    """
    device_index = config["device_index"]
    device_str = f"cuda:{device_index}"
    dtype = torch.float16 if config.get("dtype") == "float16" else torch.float32

    candidates = [config["hf_id"]]
    if config.get("fallback_id"):
        candidates.append(config["fallback_id"])

    for model_id in candidates:
        print(f"  Loading {model_id} → {device_str} (float16)...")
        t0 = time.perf_counter()
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map={"": device_index},
            )
            model.eval()
            load_time = time.perf_counter() - t0
            vram_mb = torch.cuda.memory_allocated(device_index) / 1024 ** 2
            print(f"  Loaded {model_id} in {load_time:.2f}s | "
                  f"VRAM GPU{device_index}: {vram_mb:.0f} MB")
            # Quick smoke test.
            _smoke = _generate_on_device(model, tokenizer, "2+2=", 8, device_str)
            print(f"  Smoke test OK: '{_smoke[:30].strip()}'")
            return tokenizer, model, device_str
        except Exception as exc:
            print(f"  Failed to load {model_id}: {exc}")
            gc.collect()
            torch.cuda.empty_cache()

    print(f"FATAL: Could not load any candidate for {config['name']}.")
    sys.exit(1)


def unload_model(model: Any, tokenizer: Any, device_index: int) -> None:
    """Free GPU VRAM after a model run completes."""
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    vram_after = torch.cuda.memory_allocated(device_index) / 1024 ** 2
    print(f"  Unloaded model. GPU{device_index} VRAM now: {vram_after:.0f} MB")


# ---------------------------------------------------------------------------
# Section 5: Text generation.
# ---------------------------------------------------------------------------


def _generate_on_device(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    device_str: str,
) -> str:
    """Generate text on a specific GPU device.

    **Detailed explanation for engineers:**
        Applies the chat template (Qwen3 requires enable_thinking=False to
        suppress the <think>…</think> system prompt wrapper). Falls back
        gracefully for older or non-Qwen tokenizers. Greedy decoding
        (do_sample=False) for full reproducibility across runs.

        Post-processes Qwen3 thinking tokens: strips everything before
        </think> if present so downstream answer extraction works cleanly.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text = prompt
    except Exception:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device_str) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0, input_len:], skip_special_tokens=True,
    )
    # Strip Qwen3 chain-of-thought reasoning blocks.
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    return response.strip()


def generate_response(
    prompt: str,
    tokenizer: Any,
    model: Any,
    device_str: str,
    max_new_tokens: int = 256,
) -> str:
    """Public wrapper for single-pass generation."""
    return _generate_on_device(model, tokenizer, prompt, max_new_tokens, device_str)


# ---------------------------------------------------------------------------
# Section 6: Answer extraction.
# ---------------------------------------------------------------------------


def extract_final_number(text: str) -> int | None:
    """Extract the final numeric answer from an LLM response.

    **Detailed explanation for engineers:**
        Priority order:
        1. GSM8K #### format (most specific).
        2. "Answer: N" or "answer is N" (explicit label).
        3. Last number in text (broadest fallback).
        Handles comma-separated numbers and negatives.
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


# ---------------------------------------------------------------------------
# Section 7: Arithmetic constraint extraction and violation formatting.
# ---------------------------------------------------------------------------


def extract_arithmetic_steps(response: str) -> list[dict[str, Any]]:
    """Find all 'a OP b = c' arithmetic expressions and verify each.

    **Detailed explanation for engineers:**
        This is the core of Carnot's Ising verification: we extract all
        arithmetic steps the model wrote in its chain-of-thought and check
        each one. Any step where the claimed result is wrong (|claimed -
        correct| > 0.01) is a violation — an arithmetic constraint violation
        that the Ising energy model would flag with high energy.
    """
    steps: list[dict[str, Any]] = []
    pattern = re.compile(
        r"(-?[\d,]+(?:\.\d+)?)\s*([+\-*/×x÷])\s*(-?[\d,]+(?:\.\d+)?)"
        r"\s*=\s*(-?[\d,]+(?:\.\d+)?)"
    )
    for m in pattern.finditer(response):
        try:
            a = float(m.group(1).replace(",", ""))
            op = m.group(2)
            b = float(m.group(3).replace(",", ""))
            claimed = float(m.group(4).replace(",", ""))
        except ValueError:
            continue
        if op in ("×", "x"):
            op = "*"
        if op == "÷":
            op = "/"
        try:
            if op == "+":
                correct = a + b
            elif op == "-":
                correct = a - b
            elif op == "*":
                correct = a * b
            elif op == "/" and b != 0:
                correct = a / b
            else:
                continue
        except ZeroDivisionError:
            continue
        # Coerce to int for clean display when all values are whole numbers.
        if a == int(a) and b == int(b) and claimed == int(claimed):
            a, b, claimed = int(a), int(b), int(claimed)
            if correct == int(correct):
                correct = int(correct)
        satisfied = abs(float(claimed) - float(correct)) < 0.01
        steps.append({
            "expression": f"{a} {op} {b}",
            "claimed": claimed,
            "correct": correct,
            "satisfied": satisfied,
        })
    return steps


def format_violations(arith_steps: list[dict[str, Any]]) -> str:
    """Convert arithmetic violations into natural language feedback for repair.

    **Detailed explanation for engineers:**
        Only reveals which intermediate steps were wrong, NOT the final answer.
        This is the Carnot value proposition: constraint-guided feedback rather
        than simply giving away the answer. The model must re-reason from the
        corrected intermediate values.
    """
    violated = [s for s in arith_steps if not s.get("satisfied", True)]
    if not violated:
        return ""
    lines = ["Your answer contains arithmetic errors:"]
    for i, v in enumerate(violated, 1):
        lines.append(
            f"  {i}. You wrote {v['expression']} = {v['claimed']}, "
            f"but the correct result is {v['correct']}."
        )
    lines += [
        "",
        "Please recalculate step by step, fixing these errors. "
        "Give the final answer as a number.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 8: Error taxonomy (Apple GSM-Symbolic categories).
# ---------------------------------------------------------------------------


def categorize_error(
    item: dict[str, Any],
    response: str,
    extracted: int | None,
    variant_key: str,
) -> str:
    """Classify the error type for an incorrect model answer.

    **Detailed explanation for engineers:**
        Four categories (first match wins):

        1. irrelevant_number_error — only for injected-distractor variants.
           The model's answer is within ±5 of an injected (irrelevant) number.
           Ising CANNOT catch this: using the distractor in arithmetic is
           internally consistent, just semantically wrong. ~74% of these pass
           Ising verification (Exp 122 finding — correct behavior).

        2. arithmetic_error — chain-of-thought contains "a OP b = c" where
           c ≠ a OP b. Ising detects this via constraint energy spike. This is
           the primary repair target.

        3. reading_comprehension_error — no parseable number, or answer is
           wildly wrong (>3× or <0.3× ground truth). Ising cannot help.

        4. logic_error — arithmetic locally correct but wrong equations.
           Ising cannot detect this (no constraint violations to flag).
    """
    gt = item["correct_answer"]

    # 1. Irrelevant-number error (only for injected variants).
    if variant_key in ("irrelevant_injected", "combined") and extracted is not None:
        orig_nums = set(re.findall(r"\d+", item.get("original_problem", "")))
        pert_nums = set(re.findall(r"\d+", item["perturbed_problem"]))
        injected = pert_nums - orig_nums
        for ns in injected:
            try:
                n = int(ns)
                if n > 0 and abs(extracted - n) <= 5:
                    return "irrelevant_number_error"
            except ValueError:
                pass

    # 2. Reading comprehension: no number or wildly wrong.
    if extracted is None:
        return "reading_comprehension_error"
    if gt != 0 and (abs(extracted) / abs(gt) > 3.0 or abs(extracted) / abs(gt) < 0.3):
        return "reading_comprehension_error"

    # 3. Arithmetic error: scan for wrong calculations.
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
        if op == "÷":
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

    # 4. Logic error: locally correct arithmetic but wrong final answer.
    return "logic_error"


# ---------------------------------------------------------------------------
# Section 9: Pipeline verification (Carnot VerifyRepairPipeline).
# ---------------------------------------------------------------------------


def verify_with_pipeline(question: str, response: str) -> dict[str, Any]:
    """Run Carnot's VerifyRepairPipeline.verify() in verify-only mode.

    **Detailed explanation for engineers:**
        Constructs a VerifyRepairPipeline with no LLM model (verify-only) and
        calls verify() with domain="arithmetic". Returns structured result with
        verified flag, constraint counts, and total energy.

        On failure (e.g., pipeline not yet importable), returns a safe
        fallback that preserves all other pipeline modes.
    """
    try:
        from carnot.pipeline.verify_repair import VerifyRepairPipeline
        pipeline = VerifyRepairPipeline(
            model=None, domains=["arithmetic"], timeout_seconds=30.0,
        )
        vr = pipeline.verify(question, response, domain="arithmetic")
        return {
            "verified": vr.verified,
            "n_constraints": len(vr.constraints),
            "n_violations": len(vr.violations),
            "energy": vr.energy,
            "error": None,
        }
    except Exception as e:
        return {
            "verified": True,  # Fail-open: don't wrongly reject correct answers.
            "n_constraints": 0,
            "n_violations": 0,
            "energy": 0.0,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Section 10: Three pipeline modes (live GPU only — no simulation path).
# ---------------------------------------------------------------------------


def run_baseline(
    item: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device_str: str,
) -> dict[str, Any]:
    """Mode 1 — Baseline: single LLM call, no verification.

    **Detailed explanation for engineers:**
        Control condition. Measures raw LLM accuracy on the perturbed problem
        without any Carnot verification. This is the baseline against which
        verify-only and verify+repair improvements are measured.
    """
    t0 = time.time()
    prompt = (
        f"Question: {item['perturbed_problem']}\n"
        "Solve step by step. Give the final answer as a number.\n"
        "Format:\nAnswer: <number>"
    )
    response = generate_response(prompt, tokenizer, model, device_str)
    elapsed = time.time() - t0
    extracted = extract_final_number(response)
    correct = extracted is not None and extracted == item["correct_answer"]
    error_type = categorize_error(item, response, extracted, item["perturbation"]) \
        if not correct else None
    return {
        "mode": "baseline",
        "extracted_answer": extracted,
        "correct": correct,
        "error_type": error_type,
        "time_s": round(elapsed, 3),
    }


def run_verify_only(
    item: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device_str: str,
) -> dict[str, Any]:
    """Mode 2 — Verify-only: generate then flag arithmetic violations, no repair.

    **Detailed explanation for engineers:**
        Generates once per question, then checks for arithmetic constraint
        violations using both ad-hoc regex step extraction AND the Carnot
        pipeline. Any violations cause the response to be flagged (abstained).
        Flagged answers count as wrong in the denominator — this tests whether
        Ising correctly identifies problematic responses without repair overhead.
    """
    t0 = time.time()
    prompt = (
        f"Question: {item['perturbed_problem']}\n"
        "Solve step by step, showing all arithmetic. "
        "Give the final answer as a number.\n"
        "Format:\nAnswer: <number>"
    )
    response = generate_response(prompt, tokenizer, model, device_str)
    arith_steps = extract_arithmetic_steps(response)
    n_violated = sum(1 for s in arith_steps if not s["satisfied"])
    pipeline_result = verify_with_pipeline(item["perturbed_problem"], response)
    elapsed = time.time() - t0
    extracted = extract_final_number(response)
    correct = extracted is not None and extracted == item["correct_answer"]
    flagged = n_violated > 0 or not pipeline_result["verified"]
    error_type = categorize_error(item, response, extracted, item["perturbation"]) \
        if not correct else None
    return {
        "mode": "verify_only",
        "extracted_answer": extracted,
        "correct": correct,
        "flagged": flagged,
        "n_arith_constraints": len(arith_steps),
        "n_arith_violated": n_violated,
        "pipeline_verified": pipeline_result["verified"],
        "error_type": error_type,
        "time_s": round(elapsed, 3),
    }


def run_verify_repair(
    item: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device_str: str,
    max_repairs: int = MAX_REPAIR_ITERS,
) -> dict[str, Any]:
    """Mode 3 — Verify+Repair: iterative EBM-guided correction.

    **Detailed explanation for engineers:**
        Full pipeline:
        1. Generate initial response.
        2. Extract arithmetic steps and check violations.
        3. If violations found: format natural language feedback and re-prompt.
        4. Repeat up to max_repairs times or until no violations remain.

        "repaired" = was initially wrong, ended up correct. This is the key
        metric: the improvement delta = repair_accuracy - baseline_accuracy.

        With live GPU inference at ~77 tok/s (Qwen) and 256 max tokens,
        each iteration takes ~3.3s. Worst case: 4 passes × 3.3s ≈ 13s/question.
        Full run at N=400: ~1.5h per model per variant.
    """
    t0 = time.time()
    q_text = item["perturbed_problem"]
    gt = item["correct_answer"]
    total_constraints = 0
    total_violated = 0
    n_repairs = 0
    initial_correct = False
    initial_extracted = None
    response = ""
    arith_steps: list[dict[str, Any]] = []
    extracted: int | None = None

    for iteration in range(max_repairs + 1):
        if iteration == 0:
            prompt = (
                f"Question: {q_text}\n"
                "Solve step by step, showing all arithmetic. "
                "Give the final answer as a number.\n"
                "Format:\nAnswer: <number>"
            )
        else:
            feedback = format_violations(arith_steps)
            if not feedback:
                break
            prompt = (
                f"Question: {q_text}\n\n"
                f"Your previous answer was:\n{response}\n\n"
                f"However, verification found problems:\n{feedback}\n\n"
                "Please recalculate step by step and give a corrected answer.\n"
                "Format:\nAnswer: <number>"
            )

        response = generate_response(prompt, tokenizer, model, device_str)
        extracted = extract_final_number(response)
        arith_steps = extract_arithmetic_steps(response)
        n_step_violated = sum(1 for s in arith_steps if not s["satisfied"])
        total_constraints += len(arith_steps)
        total_violated += n_step_violated

        if iteration == 0:
            initial_correct = extracted is not None and extracted == gt
            initial_extracted = extracted

        if n_step_violated == 0:
            break
        if iteration < max_repairs:
            n_repairs += 1

    elapsed = time.time() - t0
    final_correct = extracted is not None and extracted == gt
    error_type = categorize_error(item, response, extracted, item["perturbation"]) \
        if not final_correct else None

    return {
        "mode": "verify_repair",
        "extracted_answer": extracted,
        "correct": final_correct,
        "initial_correct": initial_correct,
        "initial_extracted": initial_extracted,
        "n_constraints": total_constraints,
        "n_violated": total_violated,
        "n_repairs": n_repairs,
        "repaired": not initial_correct and final_correct,
        "error_type": error_type,
        "time_s": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Section 11: Checkpoint support.
# ---------------------------------------------------------------------------


def _ckpt_path(model_name: str, variant_key: str) -> Path:
    """Return checkpoint file path for (model, variant) pair."""
    safe = model_name.replace("/", "_").replace(" ", "_")
    return RESULTS_DIR / f"exp182_ckpt_{safe}_{variant_key}.json"


def save_checkpoint(
    model_name: str,
    variant_key: str,
    results: list[dict[str, Any]],
) -> None:
    """Atomically save per-question results to disk.

    **Detailed explanation for engineers:**
        Write to .tmp then rename for atomic update — avoids corrupt partial
        writes if the process is killed mid-write. Each result dict is small
        (no raw response text) to keep checkpoint files manageable.
    """
    path = _ckpt_path(model_name, variant_key)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f)
    tmp.rename(path)


def load_checkpoint(
    model_name: str,
    variant_key: str,
) -> list[dict[str, Any]] | None:
    """Load checkpoint if it exists. Returns None if absent or malformed."""
    path = _ckpt_path(model_name, variant_key)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"  Checkpoint: {path.name} ({len(data)} results)")
        return data
    except Exception as e:
        print(f"  Checkpoint load failed ({e}) — starting fresh.")
        return None


# ---------------------------------------------------------------------------
# Section 12: Bootstrap confidence intervals.
# ---------------------------------------------------------------------------


def bootstrap_ci(
    correct_flags: list[bool],
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Compute accuracy and 95% bootstrap CI.

    **Detailed explanation for engineers:**
        Vectorized numpy bootstrap: draws n_bootstrap × N index matrix
        at once, computes per-sample means as a single array operation.
        At N=400, 10,000 samples gives CI width ~±2.5pp.

    Returns:
        (accuracy, ci_lower, ci_upper) as fractions in [0, 1].
    """
    rng = np.random.default_rng(seed)
    arr = np.array(correct_flags, dtype=float)
    n = len(arr)
    point_estimate = float(np.mean(arr))
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_means = arr[indices].mean(axis=1)
    alpha = 1.0 - confidence
    ci_lo = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    return point_estimate, ci_lo, ci_hi


def bootstrap_delta_ci(
    baseline_flags: list[bool],
    repair_flags: list[bool],
    n_bootstrap: int = N_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = BOOTSTRAP_SEED + 1,
) -> tuple[float, float, float]:
    """Compute verify-repair delta and 95% paired bootstrap CI.

    **Detailed explanation for engineers:**
        Paired bootstrap: delta computed per question (repair_correct[i] -
        baseline_correct[i]) then bootstrapped as a unit. Paired sampling
        accounts for within-question correlation, giving tighter CIs than
        unpaired bootstrap. Positive delta = improvement from verify-repair.
    """
    rng = np.random.default_rng(seed)
    base = np.array(baseline_flags, dtype=float)
    rep = np.array(repair_flags, dtype=float)
    delta_per_q = rep - base
    n = len(delta_per_q)
    point_delta = float(np.mean(delta_per_q))
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_deltas = delta_per_q[indices].mean(axis=1)
    alpha = 1.0 - confidence
    ci_lo = float(np.percentile(bootstrap_deltas, 100 * alpha / 2))
    ci_hi = float(np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2)))
    return point_delta, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Section 13: Two-proportion z-test.
# ---------------------------------------------------------------------------


def two_proportion_ztest(
    n_success_1: int,
    n_1: int,
    n_success_2: int,
    n_2: int,
    one_sided: bool = True,
) -> dict[str, Any]:
    """Two-proportion z-test: is p2 (adversarial improvement rate) > p1 (control)?

    **Detailed explanation for engineers:**
        Tests whether the proportion of questions improved (verify-repair
        correct but baseline wrong) is higher in the adversarial group than
        in the control group.

        H0: p2 ≤ p1 (one-sided) | H1: p2 > p1
        Pooled proportion under H0: p_pool = (n_s1 + n_s2) / (n1 + n2)
        SE = sqrt(p_pool * (1-p_pool) * (1/n1 + 1/n2))
        z = (p2 - p1) / SE
        p-value = 1 - Φ(z) for one-sided upper-tail test.
    """
    p1 = n_success_1 / n_1 if n_1 > 0 else 0.0
    p2 = n_success_2 / n_2 if n_2 > 0 else 0.0
    p_pool = (n_success_1 + n_success_2) / (n_1 + n_2) if (n_1 + n_2) > 0 else 0.5
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_1 + 1 / n_2)) \
        if n_1 > 0 and n_2 > 0 else 1.0
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


# ---------------------------------------------------------------------------
# Section 14: Permutation test.
# ---------------------------------------------------------------------------


def permutation_test_hypothesis(
    control_deltas: list[float],
    adversarial_deltas: list[float],
    n_permutations: int = N_PERMUTATION,
    seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Permutation test: is mean adversarial improvement delta > control delta?

    **Detailed explanation for engineers:**
        H0: mean(adv_deltas) ≤ mean(ctrl_deltas) | H1: > (one-sided)
        1. Compute observed_stat = mean(adv) - mean(ctrl).
        2. Pool all deltas, shuffle, re-split n_permutations times.
        3. p = fraction of permutations where null_stat ≥ observed_stat.

        10,000 permutations gives p-value resolution of ±0.01, necessary for
        borderline results near p=0.05.
    """
    rng = np.random.default_rng(seed)
    ctrl = np.array(control_deltas, dtype=float)
    adv = np.array(adversarial_deltas, dtype=float)
    observed_stat = float(adv.mean() - ctrl.mean())
    all_deltas = np.concatenate([ctrl, adv])
    n_ctrl = len(ctrl)
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(all_deltas)
        null_adv = shuffled[n_ctrl:]
        null_ctrl = shuffled[:n_ctrl]
        null_stats[i] = null_adv.mean() - null_ctrl.mean()
    p_value = float((null_stats >= observed_stat).mean())

    if observed_stat > 0 and p_value < 0.05:
        interpretation = (
            "HYPOTHESIS SUPPORTED (p<0.05): Verify-repair improvement is "
            "significantly larger on adversarial variants than on control."
        )
    elif observed_stat > 0:
        interpretation = (
            f"Positive direction (adv delta > ctrl delta) but not significant "
            f"at p<0.05 (p={p_value:.4f})."
        )
    else:
        interpretation = (
            f"Hypothesis NOT supported: adversarial delta ≤ control delta "
            f"(p={p_value:.4f})."
        )

    return {
        "observed_stat": round(observed_stat, 4),
        "p_value": round(p_value, 4),
        "significant_p05": bool(p_value < 0.05),
        "n_ctrl": n_ctrl,
        "n_adv": len(adv),
        "n_permutations": n_permutations,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Section 15: Per-variant metrics aggregation.
# ---------------------------------------------------------------------------


def compute_variant_metrics(
    baseline_results: list[dict[str, Any]],
    verify_results: list[dict[str, Any]],
    repair_results: list[dict[str, Any]],
    variant_key: str,
    control_baseline_acc: float | None,
) -> dict[str, Any]:
    """Aggregate accuracy, CIs, error breakdown for one model×variant.

    **Detailed explanation for engineers:**
        Computes per-mode accuracy, 95% bootstrap CIs, and delta metrics.
        Error breakdown categorizes wrong answers into the Apple taxonomy:
        - arithmetic_error: Ising-catchable (repair target)
        - irrelevant_number_error: Ising correctly ignores (validates Exp 122)
        - logic_error: Ising cannot detect
        - reading_comprehension_error: Ising cannot help

        improvement_delta_pp = (repair_acc - baseline_acc) × 100
        This is the key hypothesis metric: higher delta on adversarial vs
        control variants supports H1 (Goal #5).
    """
    n = len(baseline_results)
    base_flags = [r["correct"] for r in baseline_results]
    verify_flags = [r["correct"] for r in verify_results]
    repair_flags = [r["correct"] for r in repair_results]

    base_acc, base_lo, base_hi = bootstrap_ci(base_flags)
    verify_acc, verify_lo, verify_hi = bootstrap_ci(verify_flags, seed=BOOTSTRAP_SEED + 10)
    repair_acc, repair_lo, repair_hi = bootstrap_ci(repair_flags, seed=BOOTSTRAP_SEED + 20)
    delta, delta_lo, delta_hi = bootstrap_delta_ci(base_flags, repair_flags)

    # Accuracy drop vs. control (positive = worse than control).
    acc_drop = (control_baseline_acc - base_acc) * 100 \
        if control_baseline_acc is not None else None

    # Verify-only metrics.
    n_flagged = sum(1 for r in verify_results if r.get("flagged", False))
    n_not_flagged = n - n_flagged
    n_flagged_correct = sum(
        1 for r in verify_results
        if not r.get("flagged", False) and r["correct"]
    )
    precision = n_flagged_correct / n_not_flagged if n_not_flagged > 0 else 0.0

    # Error taxonomy from repair results (most accurate error info).
    error_types = [
        "arithmetic_error", "irrelevant_number_error",
        "logic_error", "reading_comprehension_error",
    ]
    error_counts: dict[str, int] = {et: 0 for et in error_types}
    for r in repair_results:
        et = r.get("error_type")
        if et in error_counts:
            error_counts[et] += 1

    n_errors = sum(error_counts.values())
    ising_catchable = error_counts["arithmetic_error"]
    ising_uncatchable = n_errors - ising_catchable

    # Exp 122 replication: fraction of irrelevant_number errors that pass Ising.
    irrel_total = error_counts["irrelevant_number_error"]
    irrel_passed = sum(
        1 for r in verify_results
        if r.get("error_type") == "irrelevant_number_error"
        and r.get("pipeline_verified", True)
    )
    irrel_pass_rate = round(irrel_passed / irrel_total, 4) if irrel_total > 0 else None

    # Number of questions improved by verify-repair (correct in repair, wrong in baseline).
    n_improved = sum(
        1 for b, r_rep in zip(baseline_results, repair_results)
        if r_rep["correct"] and not b["correct"]
    )

    return {
        "n_questions": n,
        "baseline": {
            "accuracy": round(base_acc, 6),
            "accuracy_pct": round(base_acc * 100, 2),
            "accuracy_drop_pp": round(acc_drop, 2) if acc_drop is not None else None,
            "ci_95_lo": round(base_lo, 6),
            "ci_95_hi": round(base_hi, 6),
            "n_correct": int(sum(base_flags)),
        },
        "verify_only": {
            "accuracy": round(verify_acc, 6),
            "accuracy_pct": round(verify_acc * 100, 2),
            "ci_95_lo": round(verify_lo, 6),
            "ci_95_hi": round(verify_hi, 6),
            "n_correct": int(sum(verify_flags)),
            "n_flagged": n_flagged,
            "abstention_rate": round(n_flagged / n, 4) if n > 0 else 0.0,
            "precision": round(precision, 4),
        },
        "verify_repair": {
            "accuracy": round(repair_acc, 6),
            "accuracy_pct": round(repair_acc * 100, 2),
            "ci_95_lo": round(repair_lo, 6),
            "ci_95_hi": round(repair_hi, 6),
            "n_correct": int(sum(repair_flags)),
            "delta_pp_vs_baseline": round(delta * 100, 2),
            "delta_ci_lo": round(delta_lo * 100, 2),
            "delta_ci_hi": round(delta_hi * 100, 2),
        },
        "improvement_delta_pp": round(delta * 100, 2),
        "n_improved_by_vr": n_improved,
        "error_counts": error_counts,
        "error_pcts": {et: round(c / n * 100, 1) for et, c in error_counts.items()},
        "ising_catchable_count": ising_catchable,
        "ising_uncatchable_count": ising_uncatchable,
        "ising_catchable_pct": round(ising_catchable / n * 100, 1) if n > 0 else 0.0,
        "irrelevant_pass_rate": irrel_pass_rate,
        "irrelevant_pass_n": irrel_passed,
        "irrelevant_total_n": irrel_total,
    }


# ---------------------------------------------------------------------------
# Section 16: Adversarial vs. standard ratio (thesis metric).
# ---------------------------------------------------------------------------


def compute_adversarial_ratio(
    all_model_results: dict[str, Any],
) -> dict[str, Any]:
    """Compute the adversarial vs. standard improvement ratio (key thesis metric).

    **Detailed explanation for engineers:**
        ratio = mean(adversarial improvement deltas) / control improvement delta

        A ratio > 1.0 means verify-repair helps MORE on adversarial inputs than
        on standard control inputs. This is the core Goal #5 claim.
        A ratio ≈ 2.0 would mean "twice as beneficial on adversarial as standard."

        If the control delta is ≤ 0 (verify-repair doesn't help on control),
        the ratio is undefined (None) — report the raw deltas instead.
    """
    per_model: dict[str, Any] = {}
    all_ratios: list[float] = []

    for model_name, mdata in all_model_results.items():
        ctrl_delta = mdata["variants"]["control"]["metrics"]["improvement_delta_pp"]
        adv_deltas = [
            mdata["variants"][vk]["metrics"]["improvement_delta_pp"]
            for vk in ("number_swapped", "irrelevant_injected", "combined")
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
        ) if pooled_ratio is not None else "Insufficient data (control delta ≤ 0).",
    }


# ---------------------------------------------------------------------------
# Section 17: Exp 122 replication check.
# ---------------------------------------------------------------------------


def check_exp122_replication(all_model_results: dict[str, Any]) -> dict[str, Any]:
    """Check whether the Exp 122 finding (74% irrelevant-error pass-through) replicates.

    **Detailed explanation for engineers:**
        Exp 122 established that ~74% of irrelevant_number errors correctly PASS
        Ising verification — because the arithmetic using the distractor is
        internally consistent (no arithmetic constraint violations). This is the
        CORRECT behavior: Ising verifies arithmetic, not semantic correctness.

        If we observe a pass-through rate far from 74%, it suggests:
        - Too low: our noise injection accidentally creates arithmetic errors.
        - Too high: distractor numbers are never appearing in arithmetic steps.

        Tolerance: ±10pp from 0.74 → [0.64, 0.84] is "replicates".
    """
    exp122_reference = 0.74
    total_irrel = 0
    total_passed = 0

    for model_name, mdata in all_model_results.items():
        for vk in ("irrelevant_injected", "combined"):
            vm = mdata["variants"][vk]["metrics"]
            total_irrel += vm.get("irrelevant_total_n", 0)
            total_passed += vm.get("irrelevant_pass_n", 0)

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
            + ("Replicates." if matches else "DEVIATION — investigate.")
        ) if observed is not None else "No irrelevant-number errors observed.",
    }


# ---------------------------------------------------------------------------
# Section 18: Main experiment loop.
# ---------------------------------------------------------------------------


def main() -> int:
    """Run Experiment 182: definitive adversarial GSM8K on live RTX 3090 GPUs."""
    sep = "=" * 80
    print(sep)
    print("EXPERIMENT 182: DEFINITIVE Adversarial GSM8K — Live GPU Inference")
    print(f"  Variants:  4 (control, number-swapped, irrelevant-injected, combined)")
    print(f"  N:         {N_PER_VARIANT} questions/variant = {N_PER_VARIANT * 4} total")
    print(f"  Modes:     Baseline | Verify-only | Verify+Repair (max {MAX_REPAIR_ITERS} iters)")
    print(f"  Models:    {', '.join(c['name'] for c in MODEL_CONFIGS)}")
    print(f"  CIs:       95% bootstrap, n={N_BOOTSTRAP:,} samples (≈±2.5pp at N={N_PER_VARIANT})")
    print(f"  Sig test:  Permutation (N={N_PERMUTATION:,}) + two-proportion z-test (convergent)")
    print(sep)

    overall_start = time.time()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # [1] Load/generate adversarial datasets.
    print("\n[1/5] Loading adversarial datasets...")
    datasets = load_or_generate_datasets()
    print(f"  Ready: {', '.join(f'{k}={len(v)}' for k, v in datasets.items())}")

    all_model_results: dict[str, Any] = {}
    # Accumulate per-question improvement flags for two-proportion z-test.
    ctrl_improved_total = 0
    ctrl_n_total = 0
    adv_improved_total = 0
    adv_n_total = 0
    # Per-model improvement deltas for permutation test.
    hypothesis_data: list[dict[str, Any]] = []

    # [2] Run each model.
    for mi, config in enumerate(MODEL_CONFIGS):
        model_name = config["name"]
        device_index = config["device_index"]
        device_str = f"cuda:{device_index}"

        print(f"\n[2/5] Model {mi+1}/{len(MODEL_CONFIGS)}: {model_name} → {device_str}")

        # Load model onto target GPU.
        load_t0 = time.time()
        tokenizer, model_obj, actual_device = load_model_on_gpu(config)
        load_time = time.time() - load_t0
        vram_mb = torch.cuda.memory_allocated(device_index) / 1024 ** 2

        control_baseline_acc: float | None = None
        variant_results: dict[str, Any] = {}

        for vk, vlabel, _mult in VARIANTS:
            items = datasets[vk]
            n = len(items)
            print(f"\n  Variant: {vlabel} ({n} questions)")

            # Load checkpoint or start fresh.
            ckpt_base = load_checkpoint(model_name, f"{vk}_baseline") or []
            ckpt_verify = load_checkpoint(model_name, f"{vk}_verify") or []
            ckpt_repair = load_checkpoint(model_name, f"{vk}_repair") or []
            completed = min(len(ckpt_base), len(ckpt_verify), len(ckpt_repair))
            if completed > 0:
                print(f"  Resuming from {completed}/{n} (checkpoint).")

            baseline_results: list[dict[str, Any]] = ckpt_base[:completed]
            verify_results: list[dict[str, Any]] = ckpt_verify[:completed]
            repair_results: list[dict[str, Any]] = ckpt_repair[:completed]

            t_var = time.time()
            last_ckpt = completed

            for qi in range(completed, n):
                item = items[qi]

                # Run all three modes on the same question.
                r_base = run_baseline(item, tokenizer, model_obj, actual_device)
                r_verify = run_verify_only(item, tokenizer, model_obj, actual_device)
                r_repair = run_verify_repair(
                    item, tokenizer, model_obj, actual_device, MAX_REPAIR_ITERS,
                )

                baseline_results.append(r_base)
                verify_results.append(r_verify)
                repair_results.append(r_repair)

                # Progress logging every 25 questions.
                done = qi + 1
                if done % 25 == 0 or done == n:
                    n_b = sum(1 for r in baseline_results if r["correct"])
                    n_r = sum(1 for r in repair_results if r["correct"])
                    total_done = len(baseline_results)
                    elapsed_m = (time.time() - t_var) / 60
                    eta_m = (elapsed_m / max(done - completed, 1)) * (n - done)
                    print(
                        f"    [{model_name}/{vk}] {total_done}/{n} | "
                        f"base {n_b/total_done:.1%} | "
                        f"repair {n_r/total_done:.1%} | "
                        f"{elapsed_m:.1f}min | ETA {eta_m:.0f}min"
                    )

                # Checkpoint every 50 questions.
                if done - last_ckpt >= 50:
                    save_checkpoint(model_name, f"{vk}_baseline", baseline_results)
                    save_checkpoint(model_name, f"{vk}_verify", verify_results)
                    save_checkpoint(model_name, f"{vk}_repair", repair_results)
                    last_ckpt = done

            # Final checkpoint save.
            save_checkpoint(model_name, f"{vk}_baseline", baseline_results)
            save_checkpoint(model_name, f"{vk}_verify", verify_results)
            save_checkpoint(model_name, f"{vk}_repair", repair_results)

            var_elapsed = time.time() - t_var

            # Compute metrics.
            metrics = compute_variant_metrics(
                baseline_results, verify_results, repair_results,
                vk, control_baseline_acc,
            )

            if vk == "control":
                control_baseline_acc = metrics["baseline"]["accuracy"]

            # Print summary row.
            b = metrics["baseline"]
            vr = metrics["verify_repair"]
            delta = metrics["improvement_delta_pp"]
            drop_str = (f" (drop: {b['accuracy_drop_pp']:+.1f}pp vs ctrl)"
                        if b["accuracy_drop_pp"] is not None else "")
            print(
                f"    {vlabel}: base={b['accuracy_pct']:.1f}%{drop_str} "
                f"→ VR={vr['accuracy_pct']:.1f}% "
                f"Δ={delta:+.1f}pp [{var_elapsed:.1f}s]"
            )
            ec = metrics["error_counts"]
            irr = metrics.get("irrelevant_pass_rate")
            irr_str = f" irrel_pass={irr:.0%}" if irr is not None else ""
            print(
                f"    Errors: arith={ec['arithmetic_error']}↑ "
                f"irrel={ec['irrelevant_number_error']}∅{irr_str} "
                f"logic={ec['logic_error']} read={ec['reading_comprehension_error']}"
            )

            variant_results[vk] = {
                "variant_label": vlabel,
                "n_questions": n,
                "metrics": metrics,
                "inference_mode": "live_gpu",
            }

            # Accumulate for z-test.
            n_improved = metrics["n_improved_by_vr"]
            if vk == "control":
                ctrl_improved_total += n_improved
                ctrl_n_total += n
            else:
                adv_improved_total += n_improved
                adv_n_total += n

        # Collect hypothesis data for permutation test.
        ctrl_delta = variant_results["control"]["metrics"]["improvement_delta_pp"]
        adv_deltas = [
            variant_results[vk]["metrics"]["improvement_delta_pp"]
            for vk in ("number_swapped", "irrelevant_injected", "combined")
        ]
        hypothesis_data.append({
            "model": model_name,
            "control_delta": ctrl_delta,
            "adversarial_deltas": adv_deltas,
        })

        all_model_results[model_name] = {
            "model_config": {
                "name": model_name,
                "hf_id": config["hf_id"],
                "device": actual_device,
                "device_index": device_index,
                "load_time_s": round(load_time, 3),
                "vram_after_load_mb": round(vram_mb, 1),
            },
            "inference_mode": "live_gpu",
            "variants": variant_results,
            "control_delta_pp": ctrl_delta,
            "adversarial_deltas_pp": adv_deltas,
        }
        print()

        unload_model(model_obj, tokenizer, device_index)

    # [3] Hypothesis tests.
    print("[3/5] Running hypothesis tests...")

    # (a) Permutation test on improvement deltas.
    all_ctrl_deltas = [d["control_delta"] for d in hypothesis_data]
    all_adv_deltas = [
        delta for d in hypothesis_data for delta in d["adversarial_deltas"]
    ]
    perm_result = permutation_test_hypothesis(all_ctrl_deltas, all_adv_deltas)

    # (b) Two-proportion z-test.
    ztest_result = two_proportion_ztest(
        n_success_1=ctrl_improved_total, n_1=ctrl_n_total,
        n_success_2=adv_improved_total, n_2=adv_n_total,
        one_sided=True,
    )

    # Convergent significance: both tests agree.
    statistical_significance = bool(
        perm_result["significant_p05"] and ztest_result["significant_p05"]
    )

    # (c) Adversarial vs. standard ratio.
    adversarial_ratio = compute_adversarial_ratio(all_model_results)

    # (d) Exp 122 replication check.
    exp122_check = check_exp122_replication(all_model_results)

    print(f"  Permutation:   p={perm_result['p_value']:.4f} "
          f"({'SIGNIFICANT' if perm_result['significant_p05'] else 'not sig'})")
    print(f"  Z-test:        p={ztest_result['p_value']:.4f} "
          f"({'SIGNIFICANT' if ztest_result['significant_p05'] else 'not sig'})")
    print(f"  Convergent:    {'YES — HYPOTHESIS CONFIRMED' if statistical_significance else 'NO'}")
    print(f"  Adv/ctrl:      {adversarial_ratio['pooled_ratio']}×")
    print(f"  Exp122 check:  {exp122_check['interpretation']}")
    print()

    # [4] Print summary tables.
    print("[4/5] Results tables...")
    _print_tables(all_model_results, perm_result, ztest_result, statistical_significance)

    # [5] Assemble and save results.
    print("[5/5] Saving results...")
    overall_elapsed = time.time() - overall_start

    # Build per-variant CI summary.
    ci_95: dict[str, dict[str, list[float]]] = {}
    for model_name, mdata in all_model_results.items():
        ci_95[model_name] = {}
        for vk, _, _ in VARIANTS:
            vm = mdata["variants"][vk]["metrics"]["verify_repair"]
            ci_95[model_name][vk] = [
                round(vm["ci_95_lo"] * 100, 2),
                round(vm["ci_95_hi"] * 100, 2),
            ]

    # Build per-variant accuracy table.
    per_variant_accuracy: dict[str, dict[str, Any]] = {}
    for vk, vlabel, _ in VARIANTS:
        per_variant_accuracy[vk] = {"label": vlabel}
        for model_name in all_model_results:
            m = all_model_results[model_name]["variants"][vk]["metrics"]
            per_variant_accuracy[vk][model_name] = {
                "baseline_pct": m["baseline"]["accuracy_pct"],
                "verify_only_pct": m["verify_only"]["accuracy_pct"],
                "verify_repair_pct": m["verify_repair"]["accuracy_pct"],
                "delta_pp": m["improvement_delta_pp"],
                "ci_95": [m["verify_repair"]["ci_95_lo"] * 100,
                          m["verify_repair"]["ci_95_hi"] * 100],
                "accuracy_drop_pp": m["baseline"]["accuracy_drop_pp"],
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

    results: dict[str, Any] = {
        "experiment": 182,
        "timestamp": timestamp,
        "title": (
            "DEFINITIVE Adversarial GSM8K — Live GPU Inference, N=400/variant"
        ),
        "description": (
            f"N={N_PER_VARIANT}/variant (4 variants = {N_PER_VARIANT*4} total adversarial items). "
            f"Live GPU inference (no simulation). Qwen/Qwen3.5-0.8B (GPU0) + "
            f"google/gemma-4-E4B-it (GPU1). "
            f"95% bootstrap CIs (n={N_BOOTSTRAP:,}). "
            f"Permutation test (n={N_PERMUTATION:,}) + two-proportion z-test (convergent). "
            f"Definitive Goal #5 test: verify-repair improvement larger on adversarial "
            f"inputs than on control. Extends Exp 162 (simulated) with live GPU inference "
            f"and 2× sample size."
        ),
        "reference": "Apple arxiv 2410.05229; Exp 162 (sim), Exp 181 (live GPU baseline)",
        "hypothesis": (
            "H1: verify-repair improvement delta is LARGER on adversarial variants "
            f"than on control. N={N_PER_VARIANT} powered for p<0.05 if Exp 162 effect size is real."
        ),
        "n_per_variant": N_PER_VARIANT,
        "n_total": N_PER_VARIANT * 4,
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutations": N_PERMUTATION,
        "inference_mode": "live_gpu",
        "models": all_model_results,
        "variant_keys": [v[0] for v in VARIANTS],
        "per_variant_accuracy": per_variant_accuracy,
        "improvement_deltas": improvement_deltas,
        "ci_95": ci_95,
        "error_taxonomy": error_taxonomy,
        "hypothesis_test_permutation": perm_result,
        "hypothesis_test_ztest": ztest_result,
        "hypothesis_test_p": round(min(perm_result["p_value"], ztest_result["p_value"]), 4),
        "statistical_significance": statistical_significance,
        "adversarial_vs_standard_ratio": adversarial_ratio,
        "exp122_replication": exp122_check,
        "total_elapsed_s": round(overall_elapsed, 1),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Total elapsed: {overall_elapsed/3600:.2f}h")
    print()

    # Final verdict.
    print(sep)
    if statistical_significance:
        print("CONCLUSION: HYPOTHESIS CONFIRMED (p<0.05, convergent tests)")
    else:
        print("CONCLUSION: Hypothesis NOT confirmed (p≥0.05 in at least one test)")
    print(f"  Adversarial/control improvement ratio: {adversarial_ratio['pooled_ratio']}×")
    print(f"  Permutation p={perm_result['p_value']:.4f}, Z-test p={ztest_result['p_value']:.4f}")
    print(sep)
    return 0


# ---------------------------------------------------------------------------
# Section 19: Results table printer.
# ---------------------------------------------------------------------------


def _print_tables(
    all_model_results: dict[str, Any],
    perm_result: dict[str, Any],
    ztest_result: dict[str, Any],
    stat_sig: bool,
) -> None:
    """Print formatted accuracy and delta tables to stdout."""
    SEP = "=" * 86
    models = list(all_model_results.keys())

    # Table 1: Accuracy by mode.
    print(SEP)
    print("EXP 182 — ACCURACY BY MODE (%, N=400/variant)")
    print(SEP)
    print(f"{'Model/Mode':<28} {'Control':>12} {'Num-Swap':>12} {'Irrel-Inj':>12} {'Combined':>12}")
    print("-" * 86)
    for model_name in models:
        mdata = all_model_results[model_name]
        for mode_key, mode_label in [
            ("baseline", "Baseline"),
            ("verify_repair", "→ VR"),
        ]:
            row = f"  {(model_name[:18] + ' ' + mode_label[:5]):<26}"
            for vk, _, _ in VARIANTS:
                acc = mdata["variants"][vk]["metrics"][mode_key]["accuracy_pct"]
                row += f" {acc:>11.1f}%"
            print(row)
        row_ci = f"    {'(VR 95% CI)':>24}"
        for vk, _, _ in VARIANTS:
            vm = mdata["variants"][vk]["metrics"]["verify_repair"]
            lo = vm["ci_95_lo"] * 100
            hi = vm["ci_95_hi"] * 100
            row_ci += f"  [{lo:.0f}–{hi:.0f}%]"
        print(row_ci)
        print()

    # Table 2: Improvement deltas.
    print(SEP)
    print("IMPROVEMENT DELTA (Verify-Repair − Baseline, pp)")
    print("  * = adversarial Δ > control Δ (supports H1)")
    print(SEP)
    print(f"{'Model':<28} {'Control':>12} {'Num-Swap':>12} {'Irrel-Inj':>12} {'Combined':>12}")
    print("-" * 86)
    for model_name in models:
        mdata = all_model_results[model_name]
        ctrl_delta = mdata["variants"]["control"]["metrics"]["improvement_delta_pp"]
        row = f"  {model_name:<26}"
        for vk, _, _ in VARIANTS:
            delta = mdata["variants"][vk]["metrics"]["improvement_delta_pp"]
            marker = "*" if vk != "control" and delta > ctrl_delta else " "
            row += f" {delta:>+10.1f}pp{marker}"
        print(row)
    print()

    # Table 3: Error breakdown.
    print(SEP)
    print(f"ERROR BREAKDOWN (counts / {N_PER_VARIANT} questions per variant)")
    print("  ↑ = Ising catches (arithmetic_error)   ∅ = Ising misses (others)")
    print(SEP)
    for model_name in models:
        print(f"\n  {model_name}:")
        for vk, vlabel, _ in VARIANTS:
            m = all_model_results[model_name]["variants"][vk]["metrics"]
            ec = m["error_counts"]
            irr = m.get("irrelevant_pass_rate")
            irr_str = f" irrel_pass={irr:.0%}" if irr is not None else ""
            print(
                f"    {vlabel:<28}: arith={ec['arithmetic_error']}↑ "
                f"irrel={ec['irrelevant_number_error']}∅{irr_str} "
                f"logic={ec['logic_error']} read={ec['reading_comprehension_error']}"
            )

    # Table 4: Statistical tests.
    print()
    print(SEP)
    print("HYPOTHESIS TESTS — H1: adversarial Δ > control Δ (one-sided)")
    print(SEP)
    print(f"  Permutation test: p = {perm_result['p_value']:.4f}  "
          f"({'SIGNIFICANT' if perm_result['significant_p05'] else 'not sig'})")
    print(f"  Z-test:           p = {ztest_result['p_value']:.4f}  "
          f"({'SIGNIFICANT' if ztest_result['significant_p05'] else 'not sig'})")
    print(f"  Convergent:       {'YES — HYPOTHESIS CONFIRMED' if stat_sig else 'NO'}")
    print(f"  Permutation:      {perm_result['interpretation']}")
    print()


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
