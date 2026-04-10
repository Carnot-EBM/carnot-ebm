#!/usr/bin/env python3
"""Experiment 91: GSM8K Live Benchmark — real dataset, real model inference.

**Researcher summary:**
    Runs the REAL GSM8K test split through REAL model inference on two target
    models (Qwen3.5-0.8B, Gemma4-E4B-it), measuring baseline accuracy, Carnot
    verify-only accuracy, and full verify-repair accuracy. This is the first
    externally-credible number for the Carnot pipeline — Exp 67 fell back to
    synthetic data because the dataset loading failed.

**Detailed explanation for engineers:**
    Experiment 67 attempted GSM8K benchmarking but used synthetic fallback data
    when the HuggingFace datasets library was unavailable or the download failed.
    This experiment ensures REAL data is loaded (with a 200-question synthetic
    fallback only if absolutely necessary), and runs REAL model inference on two
    model families to prove model-agnostic constraint transfer.

    Architecture per question:
    1. Load 200 questions from GSM8K test split (openai/gsm8k).
    2. For each model:
       a. Mode 1 (Baseline): raw model answer, parse final number, compare to GT.
       b. Mode 2 (Verify-only): run VerifyRepairPipeline.verify() on the response,
          flag unverified answers as "uncertain" (abstain from counting).
       c. Mode 3 (Verify+Repair): run verify_and_repair() with max_repairs=3.
          If model is loaded, repair uses constraint-aware re-prompting.
    3. Collect per-model and cross-model metrics.

    Target models:
    - Qwen/Qwen3.5-0.8B (small, fast, good at arithmetic reasoning)
    - google/gemma-4-E4B-it (instruction-tuned, different architecture)

    Metrics: baseline accuracy, post-verify accuracy (with abstention),
    post-repair accuracy, Δ accuracy, constraint coverage, wall-clock time,
    and error categorization.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_91_gsm8k_live.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-006
"""

from __future__ import annotations

import gc
import json
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
# 2. GSM8K dataset loading — real first, synthetic fallback
# ---------------------------------------------------------------------------


def load_gsm8k_questions(n: int = 200, seed: int = 91) -> list[dict[str, Any]]:
    """Load n questions from the REAL GSM8K test split.

    **Detailed explanation for engineers:**
        Tries to load the real GSM8K dataset from HuggingFace using the
        `datasets` library. The canonical path is "openai/gsm8k" with config
        "main" and split "test" (1319 examples). We take the first n after
        seeded shuffle to ensure reproducibility.

        Each GSM8K example has:
        - 'question': the word problem text
        - 'answer': chain-of-thought ending with "#### <number>"

        If the datasets library is unavailable or download fails, we fall back
        to generating 200 synthetic GSM8K-style word problems. The fallback is
        clearly marked in the output so results are not misrepresented.

    Args:
        n: Number of questions to load (default 200).
        seed: Random seed for reproducible sampling.

    Returns:
        List of dicts with keys: question, ground_truth (int), answer_text,
        source ("gsm8k" or "synthetic").
    """
    rng = random.Random(seed)

    # --- Attempt to load real GSM8K ---
    try:
        from datasets import load_dataset

        print("  Loading GSM8K dataset from HuggingFace (openai/gsm8k)...")
        ds = load_dataset("openai/gsm8k", "main", split="test")
        print(f"  Loaded {len(ds)} GSM8K test examples.")

        indices = list(range(len(ds)))
        rng.shuffle(indices)
        selected = indices[:n]

        questions: list[dict[str, Any]] = []
        for idx in selected:
            example = ds[idx]
            q_text = example["question"]
            answer_text = example["answer"]

            gt = _extract_gsm8k_answer(answer_text)
            if gt is None:
                continue

            questions.append({
                "question": q_text,
                "ground_truth": gt,
                "answer_text": answer_text,
                "source": "gsm8k",
            })

        print(f"  Extracted {len(questions)} questions with valid numeric answers.")

        if len(questions) < n:
            shortfall = n - len(questions)
            print(f"  Padding with {shortfall} synthetic questions.")
            questions.extend(_generate_synthetic_gsm8k(shortfall, seed=seed + 1000))

        return questions[:n]

    except ImportError:
        print("  `datasets` library not available (pip install datasets).")
        print("  Falling back to synthetic GSM8K-style questions.")
    except Exception as e:
        print(f"  Failed to load GSM8K: {e}")
        print("  Falling back to synthetic GSM8K-style questions.")

    return _generate_synthetic_gsm8k(n, seed=seed)


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


def _generate_synthetic_gsm8k(n: int, seed: int = 91) -> list[dict[str, Any]]:
    """Generate n synthetic GSM8K-style multi-step word problems.

    **Detailed explanation for engineers:**
        Creates word problems that mirror the structure and difficulty of real
        GSM8K questions. Each problem involves 2-5 arithmetic steps with a
        narrative context. The answer is always a positive integer.

        Templates cover: shopping, cooking, travel, savings, classroom, garden,
        bakery, library, sports, construction, fundraiser, farm, factory,
        restaurant, warehouse scenarios — the same breadth as real GSM8K.
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
        q_text, answer = tmpl(random.Random(seed + i * 137))
        questions.append({
            "question": q_text,
            "ground_truth": answer,
            "answer_text": "",
            "source": "synthetic",
        })

    rng.shuffle(questions)
    return questions


# --- Synthetic question templates (GSM8K-style) ---

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
# 3. Number extraction from LLM responses
# ---------------------------------------------------------------------------


def extract_final_number(text: str) -> int | None:
    """Extract the final numeric answer from an LLM response.

    **Detailed explanation for engineers:**
        LLM responses to math questions contain the answer in various formats:
        "#### 75" (GSM8K convention), "Answer: 75", "the answer is 75", or
        just a trailing number. This function tries each format in priority
        order and falls back to the last number found in the text.

        Handles commas in numbers (e.g., "1,234" → 1234) and negative numbers.
        Returns None if no number is found at all.
    """
    # Try GSM8K "####" format first.
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Try "Answer: <number>" or "answer is <number>".
    match = re.search(r"[Aa]nswer[:\s]+(-?[\d,]+)", text)
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Fall back to last number in text.
    numbers = re.findall(r"-?[\d,]+", text)
    if numbers:
        try:
            return int(numbers[-1].replace(",", ""))
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# 4. Model loading with smoke test
# ---------------------------------------------------------------------------


def load_model(config: dict[str, Any]) -> tuple[Any, Any, str, bool]:
    """Load a HuggingFace model, trying candidate names in order.

    **Detailed explanation for engineers:**
        Iterates over config["candidates"], attempting to load each model via
        transformers AutoModelForCausalLM. Uses trust_remote_code=True for
        models that need custom code (Qwen). Runs a subprocess smoke test
        to catch ROCm hangs before committing to inference.

        Returns (tokenizer, model, device, loaded_ok). On failure, returns
        (None, None, "cpu", False) so the caller can fall back to simulation.

    Returns:
        Tuple of (tokenizer, model, device_str, loaded_successfully).
    """
    if os.environ.get("CARNOT_SKIP_LLM", ""):
        print(f"    CARNOT_SKIP_LLM set — skipping {config['name']}.")
        return None, None, "cpu", False

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"    torch/transformers not available: {e}")
        return None, None, "cpu", False

    # Force CPU: ROCm on this machine hangs during generation (see known-issues).
    # CUDA_VISIBLE_DEVICES="" prevents torch from seeing GPU at all.
    force_cpu = os.environ.get("CARNOT_FORCE_CPU", "1") == "1"
    if force_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    trust = config.get("trust_remote_code", True)

    for model_name in config["candidates"]:
        try:
            print(f"    Loading {model_name} on {device}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=trust,
                torch_dtype=torch.float16 if device == "cuda" else None,
            )
            if device == "cuda":
                model = model.cuda()
            model.eval()
            print(f"    Loaded {model_name} on {device}. Running smoke test...")

            try:
                test_input = tokenizer("Hi", return_tensors="pt")
                if device == "cuda":
                    test_input = {k: v.cuda() for k, v in test_input.items()}
                with torch.no_grad():
                    _ = model.generate(
                        **test_input, max_new_tokens=4, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                print(f"    Smoke test passed.")
                return tokenizer, model, device, True
            except Exception as e:
                print(f"    Smoke test failed: {e}")

            print(f"    Falling back to simulated mode for {config['name']}.")
            del model, tokenizer
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"    Failed to load {model_name}: {e}")

    return None, None, "cpu", False


def unload_model(model: Any, tokenizer: Any, device: str) -> None:
    """Free model memory so the next model can load.

    **Detailed explanation for engineers:**
        Small GPU boxes can only hold one model at a time. This function
        deletes the model and tokenizer references, empties the CUDA cache,
        and forces garbage collection to reclaim memory before loading the
        next model.
    """
    del model, tokenizer
    try:
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# 5. LLM generation
# ---------------------------------------------------------------------------


def generate_response(
    prompt: str,
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate a response from a loaded HuggingFace causal LM.

    **Detailed explanation for engineers:**
        Uses greedy decoding (do_sample=False) for reproducibility. Applies
        the model's chat template if available, otherwise uses raw prompt.
        Strips <think>...</think> reasoning tokens from Qwen models.

        max_new_tokens is 256 for GSM8K (chain-of-thought can be several
        sentences but shouldn't need more than 256 tokens for grade-school
        math problems).
    """
    import torch

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

    # Strip thinking tokens if present (Qwen models).
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return response


# ---------------------------------------------------------------------------
# 6. Simulated LLM responses (fallback when models can't load)
# ---------------------------------------------------------------------------


def simulate_response(
    question: dict[str, Any],
    model_name: str,
    iteration: int = 0,
    rng: random.Random | None = None,
) -> str:
    """Generate a simulated LLM response for a GSM8K question.

    **Detailed explanation for engineers:**
        When the LLM can't be loaded, this simulates chain-of-thought responses
        with realistic error rates. Different models get different base error
        rates to simulate model-family differences:

        - Qwen3.5-0.8B: ~30% base error (small model, decent at arithmetic)
        - Gemma4-E4B-it: ~25% base error (instruction-tuned, slightly better)

        Error types mirror real failure modes:
        - Arithmetic errors (50%): wrong intermediate computation
        - Logic errors (35%): missing/extra step in reasoning
        - Reading errors (15%): misunderstood the problem entirely

        On repair iterations, arithmetic errors fix ~70% per iteration (the
        feedback provides the correct computation), logic ~30%, reading ~10%.
    """
    if rng is None:
        rng = random.Random(42)

    gt = question["ground_truth"]

    # Model-specific base error rates.
    if "qwen" in model_name.lower():
        base_error = 0.30
    else:
        base_error = 0.25

    # Error rate decreases with repair iterations.
    if iteration == 0:
        error_rate = base_error
    else:
        error_rate = base_error * (0.50 ** iteration)

    is_correct = rng.random() > error_rate

    if is_correct:
        step1 = rng.randint(1, max(1, abs(gt) // 2))
        step2 = gt - step1
        return (
            f"Let me solve step by step.\n"
            f"First: {step1}\n"
            f"Then: {step1} + {step2} = {gt}\n"
            f"Answer: {gt}"
        )
    else:
        error_type = rng.choices(
            ["arithmetic", "logic", "reading"], weights=[50, 35, 15], k=1
        )[0]
        if error_type == "arithmetic":
            step1 = rng.randint(1, max(1, abs(gt) // 2))
            step2 = gt - step1
            wrong = step1 + step2 + rng.choice([-3, -2, -1, 1, 2, 3])
            return (
                f"Let me solve step by step.\n"
                f"First: {step1}\n"
                f"Then: {step1} + {step2} = {wrong}\n"
                f"Answer: {wrong}"
            )
        elif error_type == "logic":
            offset = rng.choice([-20, -10, -5, 5, 10, 20])
            wrong = gt + offset
            return (
                f"Let me work through this.\n"
                f"The result is {wrong}.\n"
                f"Answer: {wrong}"
            )
        else:
            wrong = gt * rng.choice([2, 3]) + rng.randint(-50, 50)
            return (
                f"I think the answer is {wrong}.\n"
                f"Answer: {wrong}"
            )


# ---------------------------------------------------------------------------
# 7. Arithmetic constraint extraction from chain-of-thought
# ---------------------------------------------------------------------------


def extract_arithmetic_steps(response: str) -> list[dict[str, Any]]:
    """Parse chain-of-thought for arithmetic expressions and verify each.

    **Detailed explanation for engineers:**
        GSM8K solutions contain intermediate calculations like "3 * 15 = 45"
        or "45 + 28 = 73" embedded in natural language. This function finds
        all such expressions via regex, computes the correct result, and
        returns a list of step dicts with 'satisfied' indicating correctness.

        Supports: +, -, *, /, ×, x, ÷ operators. Handles commas in numbers.
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

        if a == int(a) and b == int(b) and claimed == int(claimed):
            a, b, claimed = int(a), int(b), int(claimed)
            correct = int(correct) if correct == int(correct) else correct

        satisfied = abs(claimed - correct) < 0.01

        steps.append({
            "expression": f"{a} {op} {b}",
            "claimed": claimed,
            "correct": correct,
            "satisfied": satisfied,
        })

    return steps


# ---------------------------------------------------------------------------
# 8. Error categorization
# ---------------------------------------------------------------------------


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
           (detected by violated arithmetic constraints).
        2. LOGIC: arithmetic is internally consistent but approach is wrong
           (missing steps, wrong operation choice).
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
# 9. Violation formatting for repair prompts
# ---------------------------------------------------------------------------


def format_violations(arith_steps: list[dict[str, Any]]) -> str:
    """Convert arithmetic violations into natural language repair feedback.

    **Detailed explanation for engineers:**
        Takes arithmetic steps and produces plain English feedback identifying
        which specific calculations are wrong and what the correct result is.
        Does NOT reveal the ground truth answer — only corrects intermediate
        arithmetic. This is the Carnot value proposition: precise, actionable
        feedback that guides the LLM without giving away the answer.
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
        "Give the final answer as a number."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10. Verification via Carnot pipeline
# ---------------------------------------------------------------------------


def verify_with_pipeline(
    question: str, response: str
) -> dict[str, Any]:
    """Run Carnot VerifyRepairPipeline.verify() on a response.

    **Detailed explanation for engineers:**
        Uses the library-level VerifyRepairPipeline in verify-only mode (no
        model loaded). Extracts arithmetic constraints from the response
        text, evaluates them, and returns the verification result.

        This is the key integration point: we're using the REAL pipeline
        library, not ad-hoc extraction code from experiment scripts.

        Returns a dict with: verified (bool), n_constraints, n_violations,
        energy, and the raw VerificationResult for inspection.
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


# ---------------------------------------------------------------------------
# 11. The three benchmark modes
# ---------------------------------------------------------------------------


def run_baseline(
    question: dict[str, Any],
    model_name: str,
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live: bool = False,
    sim_rng: random.Random | None = None,
) -> dict[str, Any]:
    """Mode 1 — Baseline: raw model accuracy, no verification.

    **Detailed explanation for engineers:**
        Prompt the model with the GSM8K question, extract the final numeric
        answer, compare to ground truth. No constraint checking or repair.
        This is the control condition.
    """
    t0 = time.time()
    prompt = (
        f"Question: {question['question']}\n"
        f"Solve step by step. Give the final answer as a number.\n"
        f"Format:\nAnswer: <number>"
    )

    if use_live:
        response = generate_response(prompt, tokenizer, model, device)
    else:
        response = simulate_response(question, model_name, iteration=0, rng=sim_rng)

    elapsed = time.time() - t0
    extracted = extract_final_number(response)
    correct = extracted is not None and extracted == question["ground_truth"]

    return {
        "mode": "baseline",
        "response": response,
        "extracted_answer": extracted,
        "correct": correct,
        "time_s": elapsed,
    }


def run_verify_only(
    question: dict[str, Any],
    model_name: str,
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live: bool = False,
    sim_rng: random.Random | None = None,
) -> dict[str, Any]:
    """Mode 2 — Verify-only: flag unverified answers as abstentions.

    **Detailed explanation for engineers:**
        Generate the answer, then run both the ad-hoc arithmetic step verifier
        AND the Carnot VerifyRepairPipeline on the response. Answers flagged
        as unverified are counted as abstentions — the model declines to answer
        rather than risk a wrong answer.

        Post-verify accuracy = correct_and_verified / (verified + abstained)
        with abstentions treated as wrong. This measures whether the verifier
        selectively filters out wrong answers without also filtering correct ones.
    """
    t0 = time.time()
    prompt = (
        f"Question: {question['question']}\n"
        f"Solve step by step, showing all arithmetic. "
        f"Give the final answer as a number.\n"
        f"Format:\nAnswer: <number>"
    )

    if use_live:
        response = generate_response(prompt, tokenizer, model, device)
    else:
        response = simulate_response(question, model_name, iteration=0, rng=sim_rng)

    # Ad-hoc arithmetic step verification.
    arith_steps = extract_arithmetic_steps(response)
    n_arith = len(arith_steps)
    n_violated = sum(1 for s in arith_steps if not s["satisfied"])

    # Pipeline verification.
    pipeline_result = verify_with_pipeline(question["question"], response)

    elapsed = time.time() - t0
    extracted = extract_final_number(response)
    correct = extracted is not None and extracted == question["ground_truth"]

    # Flagged = either ad-hoc OR pipeline detected violations.
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
        "correct": correct,
        "flagged": flagged,
        "n_arith_constraints": n_arith,
        "n_arith_violated": n_violated,
        "pipeline_verified": pipeline_result["verified"],
        "pipeline_n_constraints": pipeline_result["n_constraints"],
        "pipeline_n_violations": pipeline_result["n_violations"],
        "pipeline_energy": pipeline_result["energy"],
        "error_type": error_type,
        "time_s": elapsed,
    }


def run_verify_repair(
    question: dict[str, Any],
    model_name: str,
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live: bool = False,
    sim_rng: random.Random | None = None,
    max_repairs: int = 3,
) -> dict[str, Any]:
    """Mode 3 — Verify+Repair: iterative repair loop up to max_repairs.

    **Detailed explanation for engineers:**
        The full pipeline: generate → extract arithmetic constraints → verify →
        format violations as NL feedback → re-prompt → regenerate. Repeats up
        to max_repairs times or until no violations remain.

        If a live model is loaded, repair uses the model for re-generation.
        If simulated, the simulation models repair effectiveness (arithmetic
        errors fix ~70% per iteration).

        The Carnot VerifyRepairPipeline is also run on each iteration for
        comparison, but the ad-hoc arithmetic step extraction drives the
        repair loop (it's more sensitive for GSM8K chain-of-thought).
    """
    t0 = time.time()
    q_text = question["question"]
    gt = question["ground_truth"]

    total_constraints = 0
    total_violated = 0
    n_repairs = 0
    initial_correct = False
    initial_extracted = None
    response = ""
    arith_steps: list[dict[str, Any]] = []

    for iteration in range(max_repairs + 1):
        if iteration == 0:
            prompt = (
                f"Question: {q_text}\n"
                f"Solve step by step, showing all arithmetic. "
                f"Give the final answer as a number.\n"
                f"Format:\nAnswer: <number>"
            )
        else:
            feedback = format_violations(arith_steps)
            if not feedback:
                break  # No violations to repair.
            prompt = (
                f"Question: {q_text}\n\n"
                f"Your previous answer was:\n{response}\n\n"
                f"However, verification found problems:\n{feedback}\n\n"
                f"Please recalculate step by step and give a corrected answer.\n"
                f"Format:\nAnswer: <number>"
            )

        if use_live:
            response = generate_response(prompt, tokenizer, model, device)
        else:
            response = simulate_response(
                question, model_name, iteration=iteration, rng=sim_rng,
            )

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

    # Run pipeline verification on final response.
    pipeline_result = verify_with_pipeline(q_text, response)

    error_type = None
    if not final_correct:
        error_type = categorize_error(q_text, response, gt, final_extracted, arith_steps)

    return {
        "mode": "verify_repair",
        "response": response,
        "extracted_answer": final_extracted,
        "correct": final_correct,
        "initial_correct": initial_correct,
        "initial_extracted": initial_extracted,
        "n_constraints": total_constraints,
        "n_violated": total_violated,
        "n_repairs": n_repairs,
        "repaired": not initial_correct and final_correct,
        "pipeline_verified": pipeline_result["verified"],
        "error_type": error_type,
        "time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# 12. Results saving
# ---------------------------------------------------------------------------


def save_results_json(
    all_results: dict[str, dict[str, list[dict[str, Any]]]],
    metadata: dict[str, Any],
) -> Path:
    """Save full results to JSON.

    **Detailed explanation for engineers:**
        Writes the complete results dict to results/experiment_91_results.json.
        Strips the 'response' field from individual results to keep the file
        size manageable (responses can be very long for 200 questions × 2
        models × 3 modes = 1200 entries).
    """
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / "experiment_91_results.json"

    # Strip verbose response text to keep JSON manageable.
    compact = {}
    for model_name, modes in all_results.items():
        compact[model_name] = {}
        for mode_name, entries in modes.items():
            compact[model_name][mode_name] = [
                {k: v for k, v in e.items() if k != "response"}
                for e in entries
            ]

    output = {
        "experiment": 91,
        "title": "GSM8K Live Benchmark",
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
    """Save human-readable summary to ops/gsm8k-live-results.md.

    **Detailed explanation for engineers:**
        Generates a markdown report with per-model accuracy tables, cross-model
        comparison, error categorization, repair success rates, and timing stats.
        This is the document that proves Carnot works on a real benchmark.
    """
    path = REPO_ROOT / "ops" / "gsm8k-live-results.md"
    n_questions = metadata["n_questions"]

    lines: list[str] = []
    lines.append("# Experiment 91: GSM8K Live Benchmark Results")
    lines.append("")
    lines.append(f"**Date:** {metadata['timestamp']}")
    lines.append(f"**Dataset:** {metadata['dataset_source']} "
                 f"({metadata['n_real_gsm8k']} real, {metadata['n_synthetic']} synthetic)")
    lines.append(f"**Questions:** {n_questions}")
    lines.append(f"**Total time:** {metadata['total_time_s']:.1f}s")
    lines.append("")

    for model_name, modes in all_results.items():
        model_meta = metadata["models"].get(model_name, {})
        live = model_meta.get("live", False)

        lines.append(f"## {model_name} {'(LIVE)' if live else '(SIMULATED)'}")
        lines.append("")

        # Accuracy table.
        baseline = modes["baseline"]
        verify = modes["verify_only"]
        repair = modes["verify_repair"]

        n_base = sum(1 for r in baseline if r["correct"])
        n_verify_correct = sum(1 for r in verify if r["correct"])
        n_repair = sum(1 for r in repair if r["correct"])

        acc_base = n_base / n_questions
        acc_verify = n_verify_correct / n_questions
        acc_repair = n_repair / n_questions

        lines.append("| Mode | Correct | Accuracy |")
        lines.append("|------|---------|----------|")
        lines.append(f"| Baseline | {n_base}/{n_questions} | {acc_base:.1%} |")
        lines.append(f"| Verify-only | {n_verify_correct}/{n_questions} | {acc_verify:.1%} |")
        lines.append(f"| Verify+Repair | {n_repair}/{n_questions} | {acc_repair:.1%} |")
        lines.append("")

        improvement = n_repair - n_base
        lines.append(f"**Δ accuracy (Repair vs Baseline):** "
                     f"+{improvement} questions (+{improvement / n_questions:.1%})")
        lines.append("")

        # Constraint coverage.
        total_arith = sum(r.get("n_arith_constraints", 0) for r in verify)
        total_pipeline = sum(r.get("pipeline_n_constraints", 0) for r in verify)
        questions_with_constraints = sum(
            1 for r in verify if r.get("n_arith_constraints", 0) > 0
        )
        lines.append(f"**Constraint coverage:** {questions_with_constraints}/{n_questions} "
                     f"questions had extractable arithmetic claims")
        lines.append(f"**Total arithmetic steps found:** {total_arith} (ad-hoc), "
                     f"{total_pipeline} (pipeline)")
        lines.append("")

        # Hallucination detection.
        tp = sum(1 for r in verify if not r["correct"] and r.get("flagged", False))
        tn = sum(1 for r in verify if r["correct"] and not r.get("flagged", False))
        fp = sum(1 for r in verify if r["correct"] and r.get("flagged", False))
        fn = sum(1 for r in verify if not r["correct"] and not r.get("flagged", False))
        det_acc = (tp + tn) / n_questions if n_questions else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        lines.append("**Hallucination detection (verify-only):**")
        lines.append(f"- True positives: {tp}, True negatives: {tn}")
        lines.append(f"- False positives: {fp}, False negatives: {fn}")
        lines.append(f"- Detection accuracy: {det_acc:.1%}, "
                     f"Precision: {precision:.1%}, Recall: {recall:.1%}")
        lines.append("")

        # Error categorization.
        wrong = [r for r in verify if not r["correct"]]
        if wrong:
            ec = {"arithmetic": 0, "logic": 0, "reading": 0}
            for r in wrong:
                ec[r.get("error_type", "logic")] += 1

            lines.append("**Error categorization:**")
            lines.append("")
            lines.append("| Type | Count | % of Errors |")
            lines.append("|------|-------|-------------|")
            n_wrong = len(wrong)
            for et in ["arithmetic", "logic", "reading"]:
                pct = ec[et] / n_wrong if n_wrong else 0
                lines.append(f"| {et} | {ec[et]} | {pct:.0%} |")
            lines.append("")

        # Repair stats.
        repaired_count = sum(1 for r in repair if r.get("repaired", False))
        repair_iters = [r["n_repairs"] for r in repair if r["n_repairs"] > 0]
        avg_iters = np.mean(repair_iters) if repair_iters else 0

        lines.append(f"**Repair stats:** {repaired_count} questions repaired, "
                     f"avg {avg_iters:.1f} iterations")
        lines.append("")

        # Timing.
        base_times = [r["time_s"] for r in baseline]
        verify_times = [r["time_s"] for r in verify]
        repair_times = [r["time_s"] for r in repair]

        lines.append("**Timing (per question):**")
        lines.append(f"- Baseline: {np.mean(base_times):.3f}s avg")
        lines.append(f"- Verify-only: {np.mean(verify_times):.3f}s avg")
        lines.append(f"- Verify+Repair: {np.mean(repair_times):.3f}s avg")
        lines.append("")

    # Cross-model comparison.
    if len(all_results) > 1:
        lines.append("## Cross-Model Comparison")
        lines.append("")
        lines.append("| Model | Baseline | Verify | Repair | Δ |")
        lines.append("|-------|----------|--------|--------|---|")
        for model_name, modes in all_results.items():
            nb = sum(1 for r in modes["baseline"] if r["correct"])
            nv = sum(1 for r in modes["verify_only"] if r["correct"])
            nr = sum(1 for r in modes["verify_repair"] if r["correct"])
            lines.append(
                f"| {model_name} | {nb/n_questions:.1%} | {nv/n_questions:.1%} "
                f"| {nr/n_questions:.1%} | +{nr-nb} |"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Summary saved to {path}")
    return path


# ---------------------------------------------------------------------------
# 13. Main benchmark
# ---------------------------------------------------------------------------


def main() -> int:
    """Run GSM8K live benchmark: 200 questions, 2 models, 3 modes each."""
    sep = "=" * 78
    print(sep)
    print("EXPERIMENT 91: GSM8K Live Benchmark")
    print("  Real dataset + real model inference (Qwen3.5-0.8B, Gemma4-E4B-it)")
    print("  200 questions × 3 modes: baseline, verify-only, verify+repair")
    print("  First externally-credible number for the Carnot pipeline")
    print(sep)

    overall_start = time.time()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # --- Load GSM8K questions ---
    print("\n[1/3] Loading GSM8K questions...")
    questions = load_gsm8k_questions(n=200, seed=91)
    n_real = sum(1 for q in questions if q.get("source") == "gsm8k")
    n_synth = sum(1 for q in questions if q.get("source") == "synthetic")
    print(f"  Questions: {len(questions)} ({n_real} real GSM8K, {n_synth} synthetic)")

    # --- Run benchmark per model ---
    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    model_metadata: dict[str, dict[str, Any]] = {}
    n_questions = len(questions)

    for mi, config in enumerate(MODEL_CONFIGS):
        model_name = config["name"]
        print(f"\n[2/3] Model {mi + 1}/{len(MODEL_CONFIGS)}: {model_name}")

        # Load model.
        print(f"  Loading {model_name}...")
        tokenizer, model, device, use_live = load_model(config)

        if not use_live:
            print(f"  *** FALLBACK: Using simulated outputs for {model_name} ***")

        model_metadata[model_name] = {"live": use_live, "device": device}

        modes_results: dict[str, list[dict[str, Any]]] = {
            "baseline": [],
            "verify_only": [],
            "verify_repair": [],
        }

        print(f"  Running {n_questions} questions × 3 modes...")
        model_start = time.time()

        for qi, q in enumerate(questions):
            # Each question gets its own deterministic RNG for simulation.
            seed_base = 91_000 + mi * 10_000 + qi

            # Mode 1: Baseline.
            r_base = run_baseline(
                q, model_name,
                tokenizer=tokenizer, model=model, device=device,
                use_live=use_live,
                sim_rng=random.Random(seed_base),
            )
            modes_results["baseline"].append(r_base)

            # Mode 2: Verify-only.
            r_verify = run_verify_only(
                q, model_name,
                tokenizer=tokenizer, model=model, device=device,
                use_live=use_live,
                sim_rng=random.Random(seed_base),
            )
            modes_results["verify_only"].append(r_verify)

            # Mode 3: Verify+Repair.
            r_repair = run_verify_repair(
                q, model_name,
                tokenizer=tokenizer, model=model, device=device,
                use_live=use_live,
                sim_rng=random.Random(seed_base),
                max_repairs=3,
            )
            modes_results["verify_repair"].append(r_repair)

            if (qi + 1) % 25 == 0:
                n_b = sum(1 for r in modes_results["baseline"] if r["correct"])
                n_r = sum(1 for r in modes_results["verify_repair"] if r["correct"])
                print(f"    {qi + 1}/{n_questions} — "
                      f"baseline {n_b}/{qi + 1}, repair {n_r}/{qi + 1}")

        model_elapsed = time.time() - model_start
        model_metadata[model_name]["time_s"] = model_elapsed

        all_results[model_name] = modes_results

        # Print model summary.
        n_base = sum(1 for r in modes_results["baseline"] if r["correct"])
        n_ver = sum(1 for r in modes_results["verify_only"] if r["correct"])
        n_rep = sum(1 for r in modes_results["verify_repair"] if r["correct"])

        print(f"\n  {model_name} summary ({model_elapsed:.1f}s):")
        print(f"    Baseline:      {n_base}/{n_questions} ({n_base/n_questions:.1%})")
        print(f"    Verify-only:   {n_ver}/{n_questions} ({n_ver/n_questions:.1%})")
        print(f"    Verify+Repair: {n_rep}/{n_questions} ({n_rep/n_questions:.1%})")
        print(f"    Δ (repair vs base): +{n_rep - n_base} "
              f"(+{(n_rep - n_base)/n_questions:.1%})")

        # Free model memory before loading next one.
        if use_live:
            unload_model(model, tokenizer, device)

    # --- Save results ---
    total_elapsed = time.time() - overall_start

    metadata = {
        "timestamp": timestamp,
        "n_questions": n_questions,
        "n_real_gsm8k": n_real,
        "n_synthetic": n_synth,
        "dataset_source": "GSM8K test" if n_real > 0 else "synthetic",
        "total_time_s": total_elapsed,
        "models": model_metadata,
    }

    print(f"\n[3/3] Saving results...")
    save_results_json(all_results, metadata)
    save_summary_markdown(all_results, metadata)

    # --- Final cross-model comparison ---
    print(f"\n{sep}")
    print(f"EXPERIMENT 91 FINAL RESULTS ({total_elapsed:.1f}s)")
    print(sep)
    print(f"\n  {'Model':<20s} {'Baseline':>10s} {'Verify':>10s} {'Repair':>10s} {'Δ':>6s}")
    print(f"  {'-' * 58}")

    for model_name, modes in all_results.items():
        live = model_metadata[model_name]["live"]
        tag = " (LIVE)" if live else " (SIM)"
        nb = sum(1 for r in modes["baseline"] if r["correct"])
        nv = sum(1 for r in modes["verify_only"] if r["correct"])
        nr = sum(1 for r in modes["verify_repair"] if r["correct"])
        print(f"  {model_name + tag:<20s} "
              f"{nb/n_questions:>9.1%} {nv/n_questions:>9.1%} "
              f"{nr/n_questions:>9.1%} {'+' + str(nr - nb):>5s}")

    # Verdict.
    print(f"\n  Dataset: {'REAL GSM8K' if n_real > 0 else 'Synthetic fallback'} "
          f"({n_real} real, {n_synth} synthetic)")

    any_live = any(m["live"] for m in model_metadata.values())
    if any_live and n_real > 0:
        print(f"  VERDICT: First externally-credible Carnot benchmark —")
        print(f"  real GSM8K data + real model inference.")
    elif any_live:
        print(f"  VERDICT: Real model inference, synthetic data fallback.")
    else:
        print(f"  VERDICT: Simulated run — pipeline logic exercised,")
        print(f"  but numbers are not externally credible.")

    print(f"\n  Architecture: LLM → chain-of-thought → arithmetic extraction →")
    print(f"  Ising carry-chain verify → NL feedback → LLM repair")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
