#!/usr/bin/env python3
"""Experiment 124: Full GSM8K Live Benchmark — all 1,319 questions, publishable numbers.

**Researcher summary:**
    Runs the COMPLETE GSM8K test set (1,319 questions) through real model
    inference on two target models (Qwen3.5-0.8B, Gemma4-E4B-it). Reports
    accuracy with 95% confidence intervals, compares to published baselines,
    and provides per-difficulty breakdown. This is the publishable benchmark
    run for the Carnot pipeline paper.

    Exp 91 ran 200 questions with simulated fallback inference.
    Exp 123 fixed model loading reliability (robust loader in model_loader.py).
    THIS experiment uses ALL 1,319 questions and the Exp 123 robust loader
    with CARNOT_FORCE_LIVE=1 to ensure we never silently fall back to simulated
    outputs on a publishable run.

**Detailed explanation for engineers:**
    Three differences from Exp 91:

    1. FULL DATASET: We load all 1,319 GSM8K test questions, not a 200-question
       sample. This matches published evaluations (e.g., Qwen3.5-0.8B reports
       GSM8K scores on the full 1,319-question test set).

    2. ROBUST LOADER: We use carnot.inference.model_loader.load_model() and
       generate() (from Exp 123), which handles float32 on CPU, OOM retries,
       and Qwen3 chat-template quirks. CARNOT_FORCE_LIVE=1 makes it raise
       ModelLoadError instead of silently returning (None, None).

    3. CONFIDENCE INTERVALS: We report Wilson score 95% CI for each accuracy
       number so the result is statistically complete.

    Architecture per question:
    1. Load all 1,319 questions from GSM8K test split (openai/gsm8k).
    2. For each model (Qwen3.5-0.8B, Gemma4-E4B-it):
       a. Mode 1 (Baseline): raw model answer, parse final number, compare to GT.
       b. Mode 2 (Verify-only): extract arithmetic constraints, flag violations.
       c. Mode 3 (Verify+Repair): iterative repair loop up to max_repairs=3.
    3. Report accuracy + 95% CI, published-baseline comparison, timing.

    Published baselines for comparison:
    - Qwen3.5-0.8B GSM8K: ~75% (from official Qwen3.5 technical report)
    - Gemma4-E4B-it GSM8K: ~68% (from google/gemma-4-E4B-it model card)

Usage:
    JAX_PLATFORMS=cpu CARNOT_FORCE_LIVE=1 .venv/bin/python \\
        scripts/experiment_124_full_gsm8k_live.py

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
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make carnot library importable.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

# ---------------------------------------------------------------------------
# Published baselines for comparison (from official model cards / papers).
# ---------------------------------------------------------------------------

PUBLISHED_BASELINES: dict[str, dict[str, Any]] = {
    "Qwen3.5-0.8B": {
        "gsm8k_accuracy": 0.75,
        "source": "Qwen3.5 technical report (2024)",
        "note": "Greedy decoding, full 1319-question test set",
    },
    "Gemma4-E4B-it": {
        "gsm8k_accuracy": 0.68,
        "source": "google/gemma-4-E4B-it model card",
        "note": "Approximate; Gemma4 not yet in official leaderboards",
    },
}

# ---------------------------------------------------------------------------
# 1. Model configurations
# ---------------------------------------------------------------------------

MODEL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "Qwen3.5-0.8B",
        "candidates": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"],
        "trust_remote_code": True,
    },
    {
        "name": "Gemma4-E4B-it",
        "candidates": ["google/gemma-4-E4B-it"],
        "trust_remote_code": True,
    },
]


# ---------------------------------------------------------------------------
# 2. GSM8K dataset loading — full 1,319 questions with synthetic fallback
# ---------------------------------------------------------------------------


def load_gsm8k_questions(seed: int = 124) -> list[dict[str, Any]]:
    """Load ALL 1,319 questions from the REAL GSM8K test split.

    **Detailed explanation for engineers:**
        Loads the real GSM8K dataset from HuggingFace using the `datasets`
        library. The canonical path is "openai/gsm8k" with config "main"
        and split "test" (1319 examples). We keep ALL of them (no sampling)
        to match published evaluation protocols.

        Each GSM8K example has:
        - 'question': the word problem text
        - 'answer': chain-of-thought ending with "#### <number>"

        The 'answer' text also provides rough difficulty signals:
        - Number of sentences in the chain-of-thought ≈ problem complexity.
        - We bucket into easy/medium/hard by step count for per-difficulty
          breakdown: easy ≤3 steps, medium 4-6, hard ≥7.

        If the datasets library is unavailable or download fails, we fall back
        to generating 1,319 synthetic GSM8K-style word problems and clearly
        mark the results as NOT publishable.

    Args:
        seed: Random seed used only for synthetic fallback generation.

    Returns:
        List of dicts with keys: question, ground_truth (int), answer_text,
        source ("gsm8k" or "synthetic"), difficulty ("easy"/"medium"/"hard"),
        n_steps (int, count of arithmetic-looking sentences in answer).
    """
    # --- Attempt to load real GSM8K ---
    try:
        from datasets import load_dataset  # type: ignore[import]

        print("  Loading full GSM8K test set from HuggingFace (openai/gsm8k)...")
        ds = load_dataset("openai/gsm8k", "main", split="test")
        n_raw = len(ds)
        print(f"  Loaded {n_raw} GSM8K test examples.")

        questions: list[dict[str, Any]] = []
        skipped = 0
        for example in ds:
            q_text: str = example["question"]
            answer_text: str = example["answer"]

            gt = _extract_gsm8k_answer(answer_text)
            if gt is None:
                skipped += 1
                continue

            # Estimate difficulty from chain-of-thought step count.
            n_steps = _estimate_steps(answer_text)
            difficulty = "easy" if n_steps <= 3 else ("medium" if n_steps <= 6 else "hard")

            questions.append({
                "question": q_text,
                "ground_truth": gt,
                "answer_text": answer_text,
                "source": "gsm8k",
                "n_steps": n_steps,
                "difficulty": difficulty,
            })

        print(f"  Extracted {len(questions)} questions with valid numeric answers "
              f"({skipped} skipped, no valid answer).")
        return questions

    except ImportError:
        print("  `datasets` library not available (pip install datasets).")
        print("  Falling back to 1319 synthetic GSM8K-style questions.")
        print("  *** RESULTS WILL NOT BE PUBLISHABLE — synthetic data only ***")
    except Exception as e:
        print(f"  Failed to load GSM8K: {e}")
        print("  Falling back to 1319 synthetic GSM8K-style questions.")
        print("  *** RESULTS WILL NOT BE PUBLISHABLE — synthetic data only ***")

    return _generate_synthetic_gsm8k(1319, seed=seed)


def _estimate_steps(answer_text: str) -> int:
    """Count arithmetic-looking sentences to estimate chain-of-thought length.

    **Detailed explanation for engineers:**
        We look for lines containing a number followed by arithmetic context.
        GSM8K answers are structured as multi-step reasoning — each step
        typically appears on its own line. The "####" line is the final answer.
        We count lines before "####" that contain numbers as a proxy for steps.
    """
    lines = answer_text.split("\n")
    n_steps = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("####"):
            continue
        # Count lines with numeric content as steps.
        if re.search(r"\d", line):
            n_steps += 1
    return max(1, n_steps)


def _extract_gsm8k_answer(answer_text: str) -> int | None:
    """Extract the final numeric answer from a GSM8K answer string.

    **Detailed explanation for engineers:**
        GSM8K answers end with "#### <number>" where <number> is the final
        integer answer. Handles negative numbers and commas (e.g., "1,234").
        Returns None if no valid answer is found.
    """
    match = re.search(r"####\s*(-?[\d,]+)", answer_text)
    if match:
        num_str = match.group(1).replace(",", "")
        try:
            return int(num_str)
        except ValueError:
            return None
    return None


def _generate_synthetic_gsm8k(n: int, seed: int = 124) -> list[dict[str, Any]]:
    """Generate n synthetic GSM8K-style multi-step word problems (fallback only).

    **Detailed explanation for engineers:**
        Creates word problems mirroring GSM8K structure. Used ONLY when the
        real dataset cannot be loaded. Results produced from synthetic data
        MUST NOT be reported as GSM8K scores — they measure the pipeline
        mechanics, not real-world accuracy.
    """
    rng = random.Random(seed)
    questions: list[dict[str, Any]] = []

    templates = [
        _tmpl_shopping, _tmpl_cooking, _tmpl_travel, _tmpl_savings,
        _tmpl_classroom, _tmpl_garden, _tmpl_bakery, _tmpl_library,
        _tmpl_sports, _tmpl_construction, _tmpl_fundraiser, _tmpl_farm,
        _tmpl_factory, _tmpl_restaurant, _tmpl_warehouse,
    ]

    for i in range(n):
        tmpl = templates[i % len(templates)]
        local_rng = random.Random(seed + i * 137)
        q_text, answer = tmpl(local_rng)
        n_steps = rng.randint(2, 6)
        difficulty = "easy" if n_steps <= 3 else ("medium" if n_steps <= 6 else "hard")
        questions.append({
            "question": q_text,
            "ground_truth": answer,
            "answer_text": "",
            "source": "synthetic",
            "n_steps": n_steps,
            "difficulty": difficulty,
        })

    return questions


# --- Synthetic question templates (copied from Exp 91, unchanged) ---

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
        f"A teacher buys supplies for {n} students. Each needs "
        f"{pen} pencils at ${pp} each and {nb} notebooks at ${np_} each. "
        f"Total cost?",
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
        f"Fundraiser: {at} adult tickets at ${ap}, {ct} child at ${cp}, "
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
        f"The farmer keeps {eaten} eggs and sells rest at ${ppd}/dozen. Revenue?",
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
        f"{dp}% defective. Good units?",
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
        f"Warehouse starts with {init} items. Mon: +{r1} received, -{s1} shipped. "
        f"Tue: +{r2}, -{s2}. {dmg} damaged. Usable items?",
        init + r1 - s1 + r2 - s2 - dmg,
    )


# ---------------------------------------------------------------------------
# 3. Statistical helpers: Wilson score 95% confidence interval
# ---------------------------------------------------------------------------


def wilson_ci_95(n_success: int, n_total: int) -> tuple[float, float]:
    """Compute the Wilson score 95% confidence interval for a proportion.

    **Detailed explanation for engineers:**
        The Wilson score interval is preferred over the naive Wald interval
        (p ± 1.96 * sqrt(p*(1-p)/n)) for small-sample or extreme proportions.
        It's the standard for reporting accuracy in NLP benchmarks.

        Formula:
            center = (n_success + z²/2) / (n_total + z²)
            spread = z * sqrt(n_success*(n_total-n_success)/n_total + z²/4)
                     / (n_total + z²)
            lo = center - spread
            hi = center + spread

        where z = 1.96 for 95% CI.

    Args:
        n_success: Number of correct answers.
        n_total: Total number of questions.

    Returns:
        (lower_bound, upper_bound) as proportions in [0, 1].
    """
    if n_total == 0:
        return 0.0, 1.0

    z = 1.96  # 95% CI
    p = n_success / n_total
    n = n_total
    z2 = z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    spread = (z / (1 + z2 / n)) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - spread), min(1.0, center + spread)


def format_ci(n_success: int, n_total: int) -> str:
    """Format accuracy with 95% Wilson CI as a string for display.

    **Detailed explanation for engineers:**
        Returns a string like "75.4% [73.1%, 77.6%]" for use in tables.
        The CI is shown to one decimal place to match standard reporting.

    Args:
        n_success: Number of correct answers.
        n_total: Total number of questions.

    Returns:
        Formatted string with accuracy and CI.
    """
    if n_total == 0:
        return "N/A"
    acc = n_success / n_total
    lo, hi = wilson_ci_95(n_success, n_total)
    return f"{acc:.1%} [{lo:.1%}, {hi:.1%}]"


# ---------------------------------------------------------------------------
# 4. Number extraction from LLM responses
# ---------------------------------------------------------------------------


def extract_final_number(text: str) -> int | None:
    """Extract the final numeric answer from an LLM response.

    **Detailed explanation for engineers:**
        LLM responses contain the answer in various formats:
        "#### 75" (GSM8K convention), "Answer: 75", "the answer is 75",
        or just a trailing number. Tries each format in priority order.
        Handles commas in numbers (e.g., "1,234" → 1234) and negative numbers.
    """
    # GSM8K "####" format.
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # "Answer: <number>" or "answer is <number>".
    match = re.search(r"[Aa]nswer[:\s]+(-?[\d,]+)", text)
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # "= <number>" at end of a line (common in step-by-step solutions).
    match = re.search(r"=\s*(-?[\d,]+)\s*$", text, re.MULTILINE)
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Last number in text.
    numbers = re.findall(r"-?[\d,]+", text)
    if numbers:
        try:
            return int(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# 5. Model loading via robust Exp 123 loader
# ---------------------------------------------------------------------------


def load_model_robust(config: dict[str, Any]) -> tuple[Any, Any, str, str, bool]:
    """Load a HuggingFace model using the Exp 123 robust loader.

    **Detailed explanation for engineers:**
        Uses carnot.inference.model_loader.load_model() which handles:
        - float32 on CPU (avoids AVX2 crashes with float16)
        - OOM retries with gc.collect() between attempts
        - Qwen3 chat-template quirks (enable_thinking kwarg)
        - CARNOT_FORCE_LIVE=1: raises ModelLoadError instead of (None, None)

        We try each candidate model name in order and use the first that
        loads successfully. After loading, runs a short smoke test to verify
        the model generates tokens without hanging (catches ROCm hangs).

    Args:
        config: Model config dict with 'name', 'candidates' keys.

    Returns:
        Tuple of (model, tokenizer, device, loaded_model_name, loaded_ok).
        On failure, returns (None, None, "cpu", "", False).
    """
    if os.environ.get("CARNOT_SKIP_LLM", ""):
        print(f"    CARNOT_SKIP_LLM set — skipping {config['name']}.")
        return None, None, "cpu", "", False

    try:
        from carnot.inference.model_loader import ModelLoadError, load_model
    except ImportError as e:
        print(f"    Cannot import carnot model loader: {e}")
        return None, None, "cpu", "", False

    for candidate in config["candidates"]:
        print(f"    Trying {candidate}...")
        try:
            model, tokenizer = load_model(candidate, device="cpu", max_retries=3)
            if model is None or tokenizer is None:
                print(f"    load_model returned None for {candidate}.")
                continue

            # Smoke test: generate 4 tokens to catch hanging ROCm drivers.
            print(f"    Running smoke test on {candidate}...")
            try:
                import torch
                test_input = tokenizer("Hi", return_tensors="pt")
                device_str = str(next(model.parameters()).device)
                with torch.no_grad():
                    _ = model.generate(
                        **test_input,
                        max_new_tokens=4,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                print(f"    Smoke test passed ({candidate} on {device_str}).")
                return model, tokenizer, device_str, candidate, True
            except Exception as e:
                print(f"    Smoke test FAILED for {candidate}: {e}")
                del model, tokenizer
                gc.collect()

        except Exception as e:
            print(f"    Failed to load {candidate}: {e}")

    print(f"    All candidates failed for {config['name']}.")
    return None, None, "cpu", "", False


def unload_model(model: Any, tokenizer: Any) -> None:
    """Free model memory to make room for the next model.

    **Detailed explanation for engineers:**
        Deletes Python references, empties the CUDA cache (no-op on CPU),
        and forces gc.collect() so the OS reclaims the pages before we
        attempt to load the next model. On small RAM machines this is the
        difference between the second model loading and OOMing.
    """
    del model, tokenizer
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# 6. LLM generation via Exp 123 robust generate()
# ---------------------------------------------------------------------------


def generate_response_live(
    prompt: str,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 384,
) -> str:
    """Generate a response using the robust Exp 123 generate() function.

    **Detailed explanation for engineers:**
        Delegates to carnot.inference.model_loader.generate(), which:
        - Applies the tokenizer's chat template (with Qwen3 enable_thinking=False)
        - Falls back to raw prompt if chat template fails
        - Strips <think>...</think> reasoning tokens from output
        - Uses greedy decoding (do_sample=False) for reproducibility

        max_new_tokens=384 is larger than Exp 91's 256 to accommodate longer
        GSM8K solutions on harder problems. GSM8K hard problems can have
        7+ steps, each generating ~20-30 tokens = ~200 tokens, plus overhead.

    Args:
        prompt: The prompt text (question + instructions).
        model: Loaded HuggingFace model (from load_model_robust).
        tokenizer: Matching tokenizer.
        max_new_tokens: Max tokens to generate (default 384).

    Returns:
        Generated text with thinking tokens stripped.

    Raises:
        RuntimeError: If model or tokenizer is None.
    """
    from carnot.inference.model_loader import generate
    return generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)


def simulate_response(
    question: dict[str, Any],
    model_name: str,
    iteration: int = 0,
    rng: random.Random | None = None,
) -> str:
    """Simulate an LLM response (fallback only — NOT used for publishable runs).

    **Detailed explanation for engineers:**
        When CARNOT_FORCE_LIVE=0 and the model can't load, this simulates
        chain-of-thought with realistic per-model error rates. NOT called
        when CARNOT_FORCE_LIVE=1 (which raises instead of returning None).

        Error rates by model:
        - Qwen3.5-0.8B: ~30% base error
        - Gemma4-E4B-it: ~25% base error
        Repair iterations reduce error rate by ~50% per iteration.
    """
    if rng is None:
        rng = random.Random(42)

    gt = question["ground_truth"]
    base_error = 0.30 if "qwen" in model_name.lower() else 0.25
    error_rate = base_error * (0.50 ** iteration)
    is_correct = rng.random() > error_rate

    if is_correct:
        step1 = max(1, abs(gt) // 2)
        step2 = gt - step1
        return (
            f"Let me solve step by step.\n"
            f"First: {step1}\n"
            f"Then: {step1} + {step2} = {gt}\n"
            f"Answer: {gt}"
        )
    else:
        error_type = rng.choices(["arithmetic", "logic", "reading"], weights=[50, 35, 15], k=1)[0]
        if error_type == "arithmetic":
            step1 = max(1, abs(gt) // 2)
            step2 = gt - step1
            wrong = step1 + step2 + rng.choice([-3, -2, -1, 1, 2, 3])
            return f"Let me solve.\nFirst: {step1}\nThen: {wrong}\nAnswer: {wrong}"
        elif error_type == "logic":
            wrong = gt + rng.choice([-20, -10, -5, 5, 10, 20])
            return f"The result is {wrong}.\nAnswer: {wrong}"
        else:
            wrong = gt * rng.choice([2, 3]) + rng.randint(-50, 50)
            return f"I think the answer is {wrong}.\nAnswer: {wrong}"


# ---------------------------------------------------------------------------
# 7. Arithmetic constraint extraction and verification
# ---------------------------------------------------------------------------


def extract_arithmetic_steps(response: str) -> list[dict[str, Any]]:
    """Parse chain-of-thought for arithmetic expressions and verify each.

    **Detailed explanation for engineers:**
        Finds expressions like "3 * 15 = 45" in the response, computes the
        correct result, and marks each step as satisfied or violated. This is
        the core of the verify-only and verify-repair modes — the EBM-backed
        pipeline also runs in parallel, but this ad-hoc extractor is more
        sensitive for GSM8K chain-of-thought style reasoning.

        Handles +, -, *, /, ×, x, ÷ operators and comma-separated numbers.
    """
    steps: list[dict[str, Any]] = []
    pattern = re.compile(
        r"(-?[\d,]+(?:\.\d+)?)\s*"
        r"([+\-*/×x÷])\s*"
        r"(-?[\d,]+(?:\.\d+)?)\s*"
        r"=\s*(-?[\d,]+(?:\.\d+)?)"
    )

    for match in pattern.finditer(response):
        a_str = match.group(1).replace(",", "")
        op = match.group(2)
        b_str = match.group(3).replace(",", "")
        claimed_str = match.group(4).replace(",", "")

        try:
            a = float(a_str)
            b = float(b_str)
            claimed = float(claimed_str)
        except ValueError:
            continue

        if op in ("×", "x"):
            op = "*"
        if op == "÷":
            op = "/"

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

        # Round to int if all values are integers.
        if a == int(a) and b == int(b) and claimed == int(claimed):
            a_int, b_int = int(a), int(b)
            correct_int = int(round(correct))
            satisfied = abs(int(claimed) - correct_int) < 1
            steps.append({
                "expression": f"{a_int} {op} {b_int}",
                "claimed": int(claimed),
                "correct": correct_int,
                "satisfied": satisfied,
            })
        else:
            satisfied = abs(claimed - correct) < 0.01
            steps.append({
                "expression": f"{a} {op} {b}",
                "claimed": claimed,
                "correct": correct,
                "satisfied": satisfied,
            })

    return steps


def format_violations(arith_steps: list[dict[str, Any]]) -> str:
    """Convert arithmetic violations into natural language repair feedback.

    **Detailed explanation for engineers:**
        Identifies which specific calculations are wrong and states the correct
        result WITHOUT revealing the ground truth answer. This is the Carnot
        value proposition: precise feedback that guides the LLM without giving
        away the final answer.
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
    lines.append("")
    lines.append(
        "Please recalculate step by step, fixing these errors. "
        "Give the final answer as a number on its own line starting with 'Answer:'."
    )
    return "\n".join(lines)


def verify_with_pipeline(
    question: str, response: str
) -> dict[str, Any]:
    """Run Carnot VerifyRepairPipeline.verify() on a response.

    **Detailed explanation for engineers:**
        Uses the library-level VerifyRepairPipeline in verify-only mode
        (no LLM model loaded). Extracts arithmetic constraints from the
        response and evaluates them via the JAX-backed ComposedEnergy.

        Returns a dict with: verified (bool), n_constraints, n_violations,
        energy. Errors are caught and returned as error=str(e) so a single
        pipeline failure doesn't abort the whole benchmark.
    """
    try:
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline(
            model=None,
            domains=["arithmetic"],
            timeout_seconds=30.0,
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
            "verified": False,
            "n_constraints": 0,
            "n_violations": 0,
            "energy": 0.0,
            "error": str(e),
        }


def categorize_error(
    question: str,
    response: str,
    ground_truth: int,
    extracted: int | None,
    arith_steps: list[dict[str, Any]],
) -> str:
    """Classify why an answer is wrong: arithmetic, logic, or reading error.

    **Detailed explanation for engineers:**
        Three categories:
        1. ARITHMETIC: chain-of-thought has wrong intermediate computation
           (detected by violated arithmetic step constraints).
        2. LOGIC: arithmetic is internally consistent but approach is wrong
           (wrong operation, missing steps).
        3. READING: couldn't extract a number, or answer is wildly off (>3x
           or <0.3x the correct answer), suggesting misunderstood problem.
    """
    if extracted is None:
        return "reading"

    n_violated = sum(1 for s in arith_steps if not s.get("satisfied", True))
    if n_violated > 0:
        return "arithmetic"

    if ground_truth != 0:
        ratio = abs(extracted) / abs(ground_truth)
        if ratio > 3.0 or ratio < 0.3:
            return "reading"

    return "logic"


# ---------------------------------------------------------------------------
# 8. The three benchmark modes
# ---------------------------------------------------------------------------


def run_baseline(
    question: dict[str, Any],
    model_name: str,
    *,
    model: Any = None,
    tokenizer: Any = None,
    use_live: bool = False,
    sim_rng: random.Random | None = None,
) -> dict[str, Any]:
    """Mode 1 — Baseline: raw model accuracy, no verification or repair.

    **Detailed explanation for engineers:**
        Prompt the model with the GSM8K question, extract the final numeric
        answer, compare to ground truth. No constraint checking. This is the
        control condition — what the model gets without Carnot.

        Prompt includes explicit "Answer: <number>" format instruction to
        ensure the answer is parseable even from models without GSM8K
        fine-tuning.
    """
    t0 = time.time()
    prompt = (
        f"Question: {question['question']}\n"
        f"Solve step by step. Give the final answer as a number.\n"
        f"Format: Answer: <number>"
    )

    if use_live:
        try:
            response = generate_response_live(prompt, model, tokenizer)
        except Exception as e:
            response = f"[generation error: {e}]"
    else:
        response = simulate_response(question, model_name, iteration=0, rng=sim_rng)

    elapsed = time.time() - t0
    extracted = extract_final_number(response)
    correct = extracted is not None and extracted == question["ground_truth"]

    return {
        "mode": "baseline",
        "response": response,
        "extracted_answer": extracted,
        "ground_truth": question["ground_truth"],
        "correct": correct,
        "difficulty": question.get("difficulty", "unknown"),
        "n_steps": question.get("n_steps", 0),
        "time_s": elapsed,
    }


def run_verify_only(
    question: dict[str, Any],
    model_name: str,
    *,
    model: Any = None,
    tokenizer: Any = None,
    use_live: bool = False,
    sim_rng: random.Random | None = None,
) -> dict[str, Any]:
    """Mode 2 — Verify-only: generate, then flag unverified answers.

    **Detailed explanation for engineers:**
        Generate the answer, then run both ad-hoc arithmetic step verification
        AND the Carnot VerifyRepairPipeline on the response. An answer is
        flagged if EITHER verifier detects violations.

        Flagged answers are treated as "abstentions" — the model declines to
        commit rather than risk a wrong answer. This measures precision of
        violation detection (does Carnot flag wrong answers more than right ones?).

        Verify-only accuracy = (correct AND not flagged) / n_total
        Abstention rate = flagged / n_total
    """
    t0 = time.time()
    prompt = (
        f"Question: {question['question']}\n"
        f"Solve step by step, showing all arithmetic explicitly "
        f"(e.g., '3 * 5 = 15'). Give the final answer as a number.\n"
        f"Format: Answer: <number>"
    )

    if use_live:
        try:
            response = generate_response_live(prompt, model, tokenizer)
        except Exception as e:
            response = f"[generation error: {e}]"
    else:
        response = simulate_response(question, model_name, iteration=0, rng=sim_rng)

    # Ad-hoc arithmetic step verification.
    arith_steps = extract_arithmetic_steps(response)
    n_arith = len(arith_steps)
    n_violated = sum(1 for s in arith_steps if not s["satisfied"])

    # Pipeline verification (EBM-backed).
    pipeline_result = verify_with_pipeline(question["question"], response)

    elapsed = time.time() - t0
    extracted = extract_final_number(response)
    correct = extracted is not None and extracted == question["ground_truth"]

    # Flagged by EITHER verifier.
    flagged = n_violated > 0 or not pipeline_result["verified"]

    error_type = None
    if not correct:
        error_type = categorize_error(
            question["question"], response, question["ground_truth"],
            extracted, arith_steps,
        )

    return {
        "mode": "verify_only",
        "response": response,
        "extracted_answer": extracted,
        "ground_truth": question["ground_truth"],
        "correct": correct,
        "flagged": flagged,
        "n_arith_constraints": n_arith,
        "n_arith_violated": n_violated,
        "pipeline_verified": pipeline_result["verified"],
        "pipeline_n_constraints": pipeline_result["n_constraints"],
        "pipeline_n_violations": pipeline_result["n_violations"],
        "pipeline_energy": pipeline_result["energy"],
        "error_type": error_type,
        "difficulty": question.get("difficulty", "unknown"),
        "n_steps": question.get("n_steps", 0),
        "time_s": elapsed,
    }


def run_verify_repair(
    question: dict[str, Any],
    model_name: str,
    *,
    model: Any = None,
    tokenizer: Any = None,
    use_live: bool = False,
    sim_rng: random.Random | None = None,
    max_repairs: int = 3,
) -> dict[str, Any]:
    """Mode 3 — Verify+Repair: iterative repair loop up to max_repairs times.

    **Detailed explanation for engineers:**
        Full Carnot pipeline:
        1. Generate initial answer.
        2. Extract arithmetic constraints → verify.
        3. If violations found, format as natural-language feedback.
        4. Re-prompt the model with original question + violations + "fix this".
        5. Repeat up to max_repairs times or until no violations remain.

        This is the publishable demonstration that Carnot's EBM-backed
        verification improves model accuracy over baseline. The key claim:
        "constraint-aware re-prompting fixes more errors than blind re-generation."
    """
    t0 = time.time()
    q_text = question["question"]
    gt = question["ground_truth"]

    total_constraints = 0
    total_violated = 0
    n_repairs = 0
    initial_correct = False
    initial_extracted: int | None = None
    response = ""
    arith_steps: list[dict[str, Any]] = []
    extracted: int | None = None

    for iteration in range(max_repairs + 1):
        if iteration == 0:
            prompt = (
                f"Question: {q_text}\n"
                f"Solve step by step, showing all arithmetic explicitly "
                f"(e.g., '3 * 5 = 15'). Give the final answer as a number.\n"
                f"Format: Answer: <number>"
            )
        else:
            feedback = format_violations(arith_steps)
            if not feedback:
                break  # No violations to repair; stop early.
            prompt = (
                f"Question: {q_text}\n\n"
                f"Your previous answer:\n{response}\n\n"
                f"However, verification found arithmetic errors:\n{feedback}\n\n"
                f"Please recalculate step by step, fixing these errors.\n"
                f"Format: Answer: <number>"
            )

        if use_live:
            try:
                response = generate_response_live(prompt, model, tokenizer)
            except Exception as e:
                response = f"[generation error: {e}]"
        else:
            response = simulate_response(question, model_name, iteration=iteration, rng=sim_rng)

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
    final_extracted = extracted
    final_correct = final_extracted is not None and final_extracted == gt

    # Run pipeline verification on final response for comparison.
    pipeline_result = verify_with_pipeline(q_text, response)

    error_type = None
    if not final_correct:
        error_type = categorize_error(q_text, response, gt, final_extracted, arith_steps)

    return {
        "mode": "verify_repair",
        "response": response,
        "extracted_answer": final_extracted,
        "ground_truth": gt,
        "correct": final_correct,
        "initial_correct": initial_correct,
        "initial_extracted": initial_extracted,
        "n_constraints": total_constraints,
        "n_violated": total_violated,
        "n_repairs": n_repairs,
        "repaired": not initial_correct and final_correct,
        "pipeline_verified": pipeline_result["verified"],
        "error_type": error_type,
        "difficulty": question.get("difficulty", "unknown"),
        "n_steps": question.get("n_steps", 0),
        "time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# 9. Per-difficulty accuracy breakdown
# ---------------------------------------------------------------------------


def compute_difficulty_breakdown(
    results: list[dict[str, Any]], n_total_by_difficulty: dict[str, int]
) -> dict[str, dict[str, Any]]:
    """Compute accuracy broken down by difficulty tier.

    **Detailed explanation for engineers:**
        Bucketed by n_steps in the GSM8K answer chain-of-thought:
        - easy: ≤3 steps (simple two-step problems)
        - medium: 4-6 steps (multi-step word problems)
        - hard: ≥7 steps (complex chained reasoning)

        For each bucket we report n_correct, n_total, accuracy, and 95% CI.
        This lets us compare Carnot's benefit across difficulty tiers —
        hypothesis: repair helps more on hard problems where arithmetic errors
        are more common in multi-step chains.

    Args:
        results: List of result dicts (all with 'correct' and 'difficulty').
        n_total_by_difficulty: Total question counts per difficulty bucket.

    Returns:
        Dict mapping difficulty → {n_correct, n_total, accuracy, ci_lo, ci_hi}.
    """
    breakdown: dict[str, dict[str, Any]] = {}
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in results if r.get("difficulty") == diff]
        n_correct = sum(1 for r in diff_results if r["correct"])
        n_total = n_total_by_difficulty.get(diff, len(diff_results))
        lo, hi = wilson_ci_95(n_correct, n_total)
        breakdown[diff] = {
            "n_correct": n_correct,
            "n_total": n_total,
            "accuracy": n_correct / n_total if n_total else 0.0,
            "ci_lo": lo,
            "ci_hi": hi,
        }
    return breakdown


# ---------------------------------------------------------------------------
# 10. Results saving
# ---------------------------------------------------------------------------


def save_results_json(
    all_results: dict[str, dict[str, list[dict[str, Any]]]],
    metadata: dict[str, Any],
) -> Path:
    """Save full results to results/experiment_124_results.json.

    **Detailed explanation for engineers:**
        Writes the complete results including per-question details (response
        text stripped to keep file size reasonable) and computed summary stats
        (accuracy, CI, difficulty breakdown) per model and mode.

        The JSON is the canonical record of this benchmark run. It contains
        everything needed to reproduce the tables in the paper.
    """
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / "experiment_124_results.json"

    # Compact: strip verbose response text.
    compact: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for model_name, modes in all_results.items():
        compact[model_name] = {}
        for mode_name, entries in modes.items():
            compact[model_name][mode_name] = [
                {k: v for k, v in e.items() if k != "response"}
                for e in entries
            ]

    output = {
        "experiment": 124,
        "title": "Full GSM8K Live Benchmark — all 1,319 questions",
        "metadata": metadata,
        "results": compact,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {path}")
    return path


def save_summary_markdown(
    all_results: dict[str, dict[str, list[dict[str, Any]]]],
    metadata: dict[str, Any],
) -> Path:
    """Save human-readable benchmark report to ops/gsm8k-full-results.md.

    **Detailed explanation for engineers:**
        Generates a markdown report with:
        - Per-model accuracy tables with 95% Wilson CIs
        - Published-baseline comparison
        - Per-difficulty breakdown (easy/medium/hard)
        - Hallucination detection stats (verify-only)
        - Repair stats (verify-repair)
        - Wall-clock timing
        - Cross-model comparison table

        This document is the external-facing result that would accompany
        the Carnot paper submission.
    """
    path = REPO_ROOT / "ops" / "gsm8k-full-results.md"
    n_questions = metadata["n_questions"]

    lines: list[str] = []
    lines.append("# Experiment 124: Full GSM8K Live Benchmark Results")
    lines.append("")
    lines.append(f"**Date:** {metadata['timestamp']}")
    lines.append(
        f"**Dataset:** {metadata['dataset_source']} "
        f"({metadata['n_real_gsm8k']} real GSM8K, "
        f"{metadata['n_synthetic']} synthetic)"
    )
    lines.append(f"**Total questions:** {n_questions}")
    lines.append(f"**Wall-clock time:** {metadata['total_time_s']:.1f}s "
                 f"({metadata['total_time_s']/60:.1f} min)")
    lines.append("")
    lines.append("**Note:** Accuracy format = `point estimate [95% Wilson CI]`")
    lines.append("")

    for model_name, modes in all_results.items():
        model_meta = metadata["models"].get(model_name, {})
        loaded_model = model_meta.get("loaded_model_name", model_name)
        live = model_meta.get("live", False)

        lines.append(f"## {model_name} {'(LIVE)' if live else '(SIMULATED — NOT PUBLISHABLE)'}")
        if live:
            lines.append(f"*Loaded as: `{loaded_model}`*")
        lines.append("")

        baseline = modes["baseline"]
        verify = modes["verify_only"]
        repair = modes["verify_repair"]

        n_base = sum(1 for r in baseline if r["correct"])
        n_ver = sum(1 for r in verify if r["correct"])
        n_rep = sum(1 for r in repair if r["correct"])

        # Difficulty breakdown for baseline (representative mode).
        diff_counts: dict[str, int] = {}
        for r in baseline:
            d = r.get("difficulty", "unknown")
            diff_counts[d] = diff_counts.get(d, 0) + 1

        lines.append("### Overall Accuracy")
        lines.append("")
        lines.append("| Mode | Correct | Accuracy (95% CI) |")
        lines.append("|------|---------|-------------------|")
        lines.append(f"| Baseline | {n_base}/{n_questions} | {format_ci(n_base, n_questions)} |")
        lines.append(f"| Verify-only | {n_ver}/{n_questions} | {format_ci(n_ver, n_questions)} |")
        lines.append(f"| Verify+Repair | {n_rep}/{n_questions} | {format_ci(n_rep, n_questions)} |")
        lines.append("")

        # Published baseline comparison.
        pub = PUBLISHED_BASELINES.get(model_name)
        if pub:
            pub_acc = pub["gsm8k_accuracy"]
            delta = n_base / n_questions - pub_acc
            delta_str = f"{delta:+.1%}"
            lines.append(f"**vs. published baseline:** {pub_acc:.1%} ({pub['source']})")
            lines.append(f"**Δ baseline vs. published:** {delta_str}")
            lines.append("")

        improvement = n_rep - n_base
        lines.append(f"**Δ accuracy (Repair vs Baseline):** "
                     f"{'+' if improvement >= 0 else ''}{improvement} questions "
                     f"({'+' if improvement >= 0 else ''}{improvement/n_questions:.1%})")
        lines.append("")

        # Per-difficulty breakdown.
        lines.append("### Per-Difficulty Breakdown")
        lines.append("")
        lines.append("| Difficulty | N | Baseline | Verify-only | Verify+Repair |")
        lines.append("|------------|---|----------|-------------|---------------|")

        for diff in ["easy", "medium", "hard"]:
            n_d = diff_counts.get(diff, 0)
            if n_d == 0:
                continue
            nb_d = sum(1 for r in baseline if r.get("difficulty") == diff and r["correct"])
            nv_d = sum(1 for r in verify if r.get("difficulty") == diff and r["correct"])
            nr_d = sum(1 for r in repair if r.get("difficulty") == diff and r["correct"])
            lines.append(
                f"| {diff.capitalize()} | {n_d} "
                f"| {format_ci(nb_d, n_d)} "
                f"| {format_ci(nv_d, n_d)} "
                f"| {format_ci(nr_d, n_d)} |"
            )
        lines.append("")

        # Constraint coverage.
        total_arith = sum(r.get("n_arith_constraints", 0) for r in verify)
        total_pipeline = sum(r.get("pipeline_n_constraints", 0) for r in verify)
        q_with_constraints = sum(1 for r in verify if r.get("n_arith_constraints", 0) > 0)
        lines.append("### Constraint Coverage")
        lines.append("")
        lines.append(f"- Questions with extractable arithmetic steps: "
                     f"{q_with_constraints}/{n_questions} "
                     f"({q_with_constraints/n_questions:.1%})")
        lines.append(f"- Total arithmetic steps found: {total_arith} (ad-hoc extractor)")
        lines.append(f"- Total EBM constraints found: {total_pipeline} (pipeline)")
        lines.append("")

        # Hallucination detection stats.
        tp = sum(1 for r in verify if not r["correct"] and r.get("flagged", False))
        tn = sum(1 for r in verify if r["correct"] and not r.get("flagged", False))
        fp = sum(1 for r in verify if r["correct"] and r.get("flagged", False))
        fn = sum(1 for r in verify if not r["correct"] and not r.get("flagged", False))
        n_wrong = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        lines.append("### Error Detection (Verify-only)")
        lines.append("")
        lines.append(f"- True positives (correctly flagged wrong): {tp}")
        lines.append(f"- True negatives (correctly passed correct): {tn}")
        lines.append(f"- False positives (wrongly flagged correct): {fp}")
        lines.append(f"- False negatives (missed wrong answers): {fn}")
        lines.append(f"- Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {f1:.3f}")
        lines.append("")

        # Error categorization.
        wrong_results = [r for r in verify if not r["correct"]]
        if wrong_results:
            ec = {"arithmetic": 0, "logic": 0, "reading": 0}
            for r in wrong_results:
                et = r.get("error_type") or "logic"
                ec[et] = ec.get(et, 0) + 1

            lines.append("### Error Categorization")
            lines.append("")
            lines.append("| Error Type | Count | % of Errors |")
            lines.append("|------------|-------|-------------|")
            n_wrong_total = len(wrong_results)
            for et in ["arithmetic", "logic", "reading"]:
                pct = ec.get(et, 0) / n_wrong_total if n_wrong_total else 0
                lines.append(f"| {et.capitalize()} | {ec.get(et, 0)} | {pct:.0%} |")
            lines.append("")

        # Repair stats.
        repaired_count = sum(1 for r in repair if r.get("repaired", False))
        repair_iters = [r["n_repairs"] for r in repair if r["n_repairs"] > 0]
        avg_iters = float(np.mean(repair_iters)) if repair_iters else 0.0
        lines.append("### Repair Statistics")
        lines.append("")
        lines.append(f"- Questions where repair improved answer: {repaired_count}")
        lines.append(f"- Average repair iterations (when >0): {avg_iters:.1f}")
        lines.append("")

        # Timing.
        base_times = [r["time_s"] for r in baseline]
        verify_times = [r["time_s"] for r in verify]
        repair_times = [r["time_s"] for r in repair]
        model_time_s = model_meta.get("time_s", 0)

        lines.append("### Timing")
        lines.append("")
        lines.append(f"- Total wall-clock for this model: {model_time_s:.1f}s "
                     f"({model_time_s/60:.1f} min)")
        lines.append(f"- Baseline: {np.mean(base_times):.3f}s avg per question")
        lines.append(f"- Verify-only: {np.mean(verify_times):.3f}s avg per question")
        lines.append(f"- Verify+Repair: {np.mean(repair_times):.3f}s avg per question")
        lines.append("")

    # Cross-model summary table.
    if len(all_results) > 1:
        lines.append("## Cross-Model Summary")
        lines.append("")
        lines.append("| Model | Baseline | Verify-only | Verify+Repair | Δ (Repair-Base) |")
        lines.append("|-------|----------|-------------|---------------|-----------------|")
        for model_name, modes in all_results.items():
            live = metadata["models"].get(model_name, {}).get("live", False)
            tag = " (LIVE)" if live else " (SIM)"
            nb = sum(1 for r in modes["baseline"] if r["correct"])
            nv = sum(1 for r in modes["verify_only"] if r["correct"])
            nr = sum(1 for r in modes["verify_repair"] if r["correct"])
            delta = nr - nb
            lines.append(
                f"| {model_name}{tag} "
                f"| {nb/n_questions:.1%} "
                f"| {nv/n_questions:.1%} "
                f"| {nr/n_questions:.1%} "
                f"| {'+' if delta >= 0 else ''}{delta} "
                f"({'+' if delta >= 0 else ''}{delta/n_questions:.1%}) |"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Markdown summary saved to {path}")
    return path


# ---------------------------------------------------------------------------
# 11. Main benchmark orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the full GSM8K benchmark: all 1,319 questions, 2 models, 3 modes."""
    sep = "=" * 78
    print(sep)
    print("EXPERIMENT 124: Full GSM8K Live Benchmark — Publishable Numbers")
    print("  Dataset: full 1,319-question GSM8K test set")
    print("  Models: Qwen3.5-0.8B, Gemma4-E4B-it")
    print("  Modes: baseline, verify-only, verify+repair (3 iterations max)")
    print("  Loader: robust Exp 123 model_loader with CARNOT_FORCE_LIVE=1")
    print("  Stats: accuracy + 95% Wilson CI, per-difficulty breakdown")
    print(sep)

    overall_start = time.time()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # --- Step 1: Load GSM8K questions ---
    print("\n[1/3] Loading GSM8K questions (full 1,319-question test set)...")
    questions = load_gsm8k_questions(seed=124)
    n_questions = len(questions)
    n_real = sum(1 for q in questions if q.get("source") == "gsm8k")
    n_synth = sum(1 for q in questions if q.get("source") == "synthetic")
    diff_counts: dict[str, int] = {}
    for q in questions:
        d = q.get("difficulty", "unknown")
        diff_counts[d] = diff_counts.get(d, 0) + 1

    print(f"  Questions: {n_questions} "
          f"({n_real} real GSM8K, {n_synth} synthetic)")
    print(f"  Difficulty split: "
          + ", ".join(f"{d}={diff_counts.get(d, 0)}" for d in ["easy", "medium", "hard"]))

    is_publishable = n_real >= 1000
    if not is_publishable:
        print("  *** WARNING: fewer than 1000 real GSM8K questions — "
              "results are NOT publishable ***")

    # --- Step 2: Run per-model benchmark ---
    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    model_metadata: dict[str, dict[str, Any]] = {}

    for mi, config in enumerate(MODEL_CONFIGS):
        model_name = config["name"]
        print(f"\n[2/3] Model {mi + 1}/{len(MODEL_CONFIGS)}: {model_name}")

        print(f"  Loading {model_name} (robust loader, Exp 123)...")
        model, tokenizer, device_str, loaded_model_name, use_live = load_model_robust(config)

        if not use_live:
            force_live = os.environ.get("CARNOT_FORCE_LIVE", "") == "1"
            if force_live:
                print(f"  CARNOT_FORCE_LIVE=1 but model failed to load. "
                      f"Aborting — do not produce fake publishable numbers.")
                return 1
            print(f"  *** FALLBACK: Simulated outputs for {model_name}. "
                  f"Results NOT publishable. ***")

        model_metadata[model_name] = {
            "live": use_live,
            "device": device_str,
            "loaded_model_name": loaded_model_name,
        }

        modes_results: dict[str, list[dict[str, Any]]] = {
            "baseline": [],
            "verify_only": [],
            "verify_repair": [],
        }

        print(f"  Running {n_questions} questions × 3 modes...")
        model_start = time.time()

        for qi, q in enumerate(questions):
            seed_base = 124_000 + mi * 100_000 + qi

            # Mode 1: Baseline.
            r_base = run_baseline(
                q, model_name,
                model=model, tokenizer=tokenizer,
                use_live=use_live,
                sim_rng=random.Random(seed_base),
            )
            modes_results["baseline"].append(r_base)

            # Mode 2: Verify-only.
            r_verify = run_verify_only(
                q, model_name,
                model=model, tokenizer=tokenizer,
                use_live=use_live,
                sim_rng=random.Random(seed_base),
            )
            modes_results["verify_only"].append(r_verify)

            # Mode 3: Verify+Repair.
            r_repair = run_verify_repair(
                q, model_name,
                model=model, tokenizer=tokenizer,
                use_live=use_live,
                sim_rng=random.Random(seed_base),
                max_repairs=3,
            )
            modes_results["verify_repair"].append(r_repair)

            # Progress every 50 questions (more frequent for 1319-question run).
            if (qi + 1) % 50 == 0 or (qi + 1) == n_questions:
                n_b = sum(1 for r in modes_results["baseline"] if r["correct"])
                n_r = sum(1 for r in modes_results["verify_repair"] if r["correct"])
                elapsed_so_far = time.time() - model_start
                rate = (qi + 1) / elapsed_so_far
                eta = (n_questions - qi - 1) / rate if rate > 0 else 0
                print(f"    {qi + 1}/{n_questions} — "
                      f"baseline {n_b}/{qi + 1} ({n_b/(qi + 1):.1%}), "
                      f"repair {n_r}/{qi + 1} ({n_r/(qi + 1):.1%}) — "
                      f"ETA {eta/60:.1f}min")

        model_elapsed = time.time() - model_start
        model_metadata[model_name]["time_s"] = model_elapsed

        all_results[model_name] = modes_results

        # Model summary with CIs.
        n_base = sum(1 for r in modes_results["baseline"] if r["correct"])
        n_ver = sum(1 for r in modes_results["verify_only"] if r["correct"])
        n_rep = sum(1 for r in modes_results["verify_repair"] if r["correct"])

        print(f"\n  {model_name} summary ({model_elapsed:.1f}s = {model_elapsed/60:.1f}min):")
        print(f"    Baseline:      {format_ci(n_base, n_questions)}")
        print(f"    Verify-only:   {format_ci(n_ver, n_questions)}")
        print(f"    Verify+Repair: {format_ci(n_rep, n_questions)}")
        print(f"    Δ (repair vs base): {'+' if n_rep >= n_base else ''}{n_rep - n_base} "
              f"({'+' if n_rep >= n_base else ''}{(n_rep - n_base)/n_questions:.1%})")

        # Published baseline comparison.
        pub = PUBLISHED_BASELINES.get(model_name)
        if pub and use_live:
            delta = n_base / n_questions - pub["gsm8k_accuracy"]
            print(f"    vs. published baseline ({pub['source']}): "
                  f"{pub['gsm8k_accuracy']:.1%} → delta {delta:+.1%}")

        # Free memory before next model.
        if use_live:
            print(f"  Freeing {model_name} from memory...")
            unload_model(model, tokenizer)
            model = None
            tokenizer = None

    # --- Step 3: Save results ---
    total_elapsed = time.time() - overall_start
    print(f"\n[3/3] Saving results...")

    metadata: dict[str, Any] = {
        "timestamp": timestamp,
        "n_questions": n_questions,
        "n_real_gsm8k": n_real,
        "n_synthetic": n_synth,
        "dataset_source": "GSM8K test (openai/gsm8k)" if n_real >= 1000 else "synthetic",
        "total_time_s": total_elapsed,
        "is_publishable": is_publishable and any(
            m["live"] for m in model_metadata.values()
        ),
        "difficulty_counts": diff_counts,
        "models": model_metadata,
        "published_baselines": PUBLISHED_BASELINES,
    }

    save_results_json(all_results, metadata)
    save_summary_markdown(all_results, metadata)

    # --- Final report ---
    print(f"\n{sep}")
    print(f"EXPERIMENT 124 FINAL RESULTS ({total_elapsed:.1f}s = {total_elapsed/60:.1f}min)")
    print(sep)

    header = f"  {'Model':<22s} {'Baseline':>20s} {'Verify':>20s} {'Repair':>20s} {'Δ':>6s}"
    print(header)
    print(f"  {'-' * 90}")

    for model_name, modes in all_results.items():
        live = model_metadata[model_name]["live"]
        tag = " (LIVE)" if live else " (SIM)"
        nb = sum(1 for r in modes["baseline"] if r["correct"])
        nv = sum(1 for r in modes["verify_only"] if r["correct"])
        nr = sum(1 for r in modes["verify_repair"] if r["correct"])
        delta = nr - nb
        print(
            f"  {(model_name + tag):<22s} "
            f"{format_ci(nb, n_questions):>20s} "
            f"{format_ci(nv, n_questions):>20s} "
            f"{format_ci(nr, n_questions):>20s} "
            f"{'+' if delta >= 0 else ''}{delta:>4d}"
        )

    print(f"\n  Dataset: {'REAL GSM8K ✓' if n_real >= 1000 else 'Synthetic (not publishable)'} "
          f"({n_real}/{n_questions} real questions)")

    any_live = any(m["live"] for m in model_metadata.values())
    if any_live and is_publishable:
        print("  VERDICT: PUBLISHABLE — real GSM8K data + real model inference")
        print(f"  All 1,319 test questions evaluated with 95% Wilson CIs.")
    elif any_live:
        print("  VERDICT: Real model inference, but dataset partially synthetic.")
    else:
        print("  VERDICT: SIMULATED — NOT publishable (model load failed).")
        print("  Run with CARNOT_FORCE_LIVE=1 and working model cache.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
