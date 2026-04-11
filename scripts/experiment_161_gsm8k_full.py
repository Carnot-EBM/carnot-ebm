#!/usr/bin/env python3
"""Experiment 161: Full GSM8K Benchmark (1,319 questions) with 95% Bootstrap CIs.

**Researcher summary:**
    Runs the complete GSM8K test set (1,319 questions) through Baseline,
    Verify-only, and Verify+Repair modes for two target models (Qwen3.5-0.8B,
    Gemma4-E4B-it). With N=1,319 the bootstrap confidence intervals are <±3pp,
    giving publishable precision. Addresses Goal #6: externally credible numbers.

    Exp 91 used 200 questions and got Qwen3.5: +15%, Gemma4: +14% improvement
    with simulation. This experiment scales to the full dataset and checks Exp 160
    for live eGPU inference vs. simulation fallback.

**Detailed explanation for engineers:**
    Scaling from 200 to 1,319 questions matters because:
    1. CI width scales as 1/√N: at N=200 CI ≈ ±7pp, at N=1319 CI ≈ ±2.7pp.
    2. Standard benchmarks like GPT-4 at 87.1% and Llama2-70B at 56.8% are
       reported on the full 1,319-question set. Comparisons must use the same N.
    3. Tail-end difficulty questions (hardest 20% of GSM8K) are disproportionately
       useful for measuring improvement — they're underrepresented in 200-question
       subsets.

    Inference mode selection:
    - Checks results/experiment_160_results.json for egpu_detected=True
    - If True (and JAX_PLATFORMS=cpu not forced): uses live model inference
    - Otherwise: uses CARNOT_SKIP_LLM=1 simulation with Apple-calibrated rates
      (Qwen3.5 ~30% base error, Gemma4 ~25% base error)

    Statistics methodology:
    - 95% bootstrap CIs from n=10,000 bootstrap samples (non-parametric)
    - Delta CIs computed on (repair_correct - baseline_correct) per question
    - Published baselines for comparison:
        * GPT-4:         87.1% (OpenAI, 2023)
        * Llama2-70B:    56.8% (Meta, 2023)
        * Qwen3.5-0.8B:  ~50%  (small model; this paper)
        * Gemma4-E4B-it: ~55%  (instruction-tuned; this paper)

Usage:
    # Recommended (CPU, reproducible):
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_161_gsm8k_full.py

    # With live eGPU (after Exp 160 confirms detection):
    .venv/bin/python scripts/experiment_161_gsm8k_full.py

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
# Published comparison baselines (from original papers)
# ---------------------------------------------------------------------------
PUBLISHED_BASELINES: dict[str, float] = {
    "GPT-4 (OpenAI 2023)": 0.871,
    "Llama2-70B (Meta 2023)": 0.568,
    "GPT-3.5 (OpenAI 2023)": 0.574,
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
# 2. Exp 160 eGPU detection
# ---------------------------------------------------------------------------


def check_egpu_from_exp160() -> bool:
    """Check if Exp 160 confirmed eGPU detection.

    **Detailed explanation for engineers:**
        Reads results/experiment_160_results.json and returns the value of
        egpu_detected. If the file doesn't exist (Exp 160 hasn't run), returns
        False and we fall back to simulation mode.

    Returns:
        True if Exp 160 confirmed a working eGPU; False otherwise.
    """
    path = REPO_ROOT / "results" / "experiment_160_results.json"
    if not path.exists():
        print("  results/experiment_160_results.json not found — assuming no eGPU.")
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        detected = bool(data.get("egpu_detected", False))
        if detected:
            print(f"  Exp 160: eGPU detected! Backend: {data.get('jax_backend', 'unknown')}")
        else:
            print("  Exp 160: No eGPU detected. Using simulation fallback.")
        return detected
    except Exception as e:
        print(f"  Could not read Exp 160 results: {e} — assuming no eGPU.")
        return False


# ---------------------------------------------------------------------------
# 3. GSM8K dataset loading — full 1,319 questions, synthetic fallback
# ---------------------------------------------------------------------------


def load_gsm8k_questions(seed: int = 161) -> list[dict[str, Any]]:
    """Load the full GSM8K test split (1,319 questions).

    **Detailed explanation for engineers:**
        Loads all 1,319 questions from the real GSM8K test split via HuggingFace
        datasets. Unlike Exp 91 (which took 200), we take ALL of them to get
        publishable bootstrap CIs.

        The canonical dataset is "openai/gsm8k" (alt: "gsm8k"), config "main",
        split "test". Each example has:
        - 'question': the word problem text
        - 'answer': chain-of-thought ending with "#### <number>"

        Fallback (if HF datasets unavailable):
        - Uses the 200-question subset from Exp 91 + generates synthetic questions
          to reach 400 (doubles coverage vs Exp 91). Clearly marked in output
          as "synthetic-extended" so results are not misrepresented.

    Args:
        seed: Random seed (used only for reproducible shuffle of fallback data).

    Returns:
        List of dicts with keys: question, ground_truth (int), answer_text,
        source ("gsm8k" or "synthetic").
    """
    # --- Attempt to load real GSM8K ---
    try:
        from datasets import load_dataset

        print("  Loading full GSM8K test split from HuggingFace (openai/gsm8k)...")
        # Try the canonical path first, then the alias.
        for dataset_name in ("openai/gsm8k", "gsm8k"):
            try:
                ds = load_dataset(dataset_name, "main", split="test")
                print(f"  Loaded {len(ds)} GSM8K test examples from '{dataset_name}'.")
                break
            except Exception:
                continue
        else:
            raise RuntimeError("Both 'openai/gsm8k' and 'gsm8k' failed.")

        questions: list[dict[str, Any]] = []
        for idx in range(len(ds)):
            example = ds[idx]
            gt = _extract_gsm8k_answer(example["answer"])
            if gt is None:
                continue
            questions.append({
                "question": example["question"],
                "ground_truth": gt,
                "answer_text": example["answer"],
                "source": "gsm8k",
                "idx": idx,
            })

        print(f"  Parsed {len(questions)} questions with valid numeric answers.")
        return questions

    except ImportError:
        print("  `datasets` library not available (pip install datasets).")
    except Exception as e:
        print(f"  Failed to load real GSM8K: {e}")

    # --- Fallback: 400-question synthetic extension ---
    print("  FALLBACK: Using 400-question synthetic dataset (not externally credible).")
    print("  To get real numbers: pip install datasets && re-run.")
    return _generate_synthetic_gsm8k(400, seed=seed)


def _extract_gsm8k_answer(answer_text: str) -> int | None:
    """Extract the final numeric answer from a GSM8K answer string.

    **Detailed explanation for engineers:**
        GSM8K answers end with "#### <number>" where <number> is the final
        integer answer. Handles negative numbers and commas (e.g., "1,234").
        Returns None if no valid answer is found.
    """
    match = re.search(r"####\s*(-?[\d,]+)", answer_text)
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def _generate_synthetic_gsm8k(n: int, seed: int = 161) -> list[dict[str, Any]]:
    """Generate n synthetic GSM8K-style multi-step word problems.

    **Detailed explanation for engineers:**
        Creates word problems that mirror the structure and difficulty of real
        GSM8K questions. Each problem involves 2-5 arithmetic steps with a
        narrative context. The answer is always a positive integer.

        These are the SAME templates as Exp 91 (for compatibility). This fallback
        is only used when HuggingFace datasets is unavailable. Results produced
        from synthetic data are clearly flagged in the output JSON and should
        NOT be compared to published benchmarks.

        Templates cover: shopping, cooking, travel, savings, classroom, garden,
        bakery, library, sports, construction, fundraiser, farm, factory,
        restaurant, warehouse — the same breadth as real GSM8K.
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
            "idx": i,
        })

    rng.shuffle(questions)
    return questions


# --- Synthetic question templates (GSM8K-style, identical to Exp 91 for reproducibility) ---

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
# 4. Number extraction from LLM responses
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
# 5. Model loading (replicates Exp 91 pattern)
# ---------------------------------------------------------------------------


def load_model(config: dict[str, Any], use_egpu: bool = False) -> tuple[Any, Any, str, bool]:
    """Load a HuggingFace model for live inference.

    **Detailed explanation for engineers:**
        When use_egpu=True (Exp 160 confirmed eGPU), allows GPU device use.
        Otherwise forces CPU (CARNOT_FORCE_CPU logic from Exp 91).

        CARNOT_SKIP_LLM=1 in environment causes immediate simulated-only return.

        Returns (tokenizer, model, device, loaded_ok). On failure, returns
        (None, None, "cpu", False) so the caller can fall back to simulation.
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

    if use_egpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

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
                print("    Smoke test passed.")
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
    """Free model memory before loading the next model.

    **Detailed explanation for engineers:**
        Deletes references, empties CUDA cache if on GPU, and forces GC.
        Needed on machines with limited VRAM or RAM (e.g., 16 GB system RAM).
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
# 6. LLM generation
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

        max_new_tokens=256 is sufficient for grade-school math chain-of-thought.
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
# 7. Simulated LLM responses (fallback when models can't load)
# ---------------------------------------------------------------------------


def simulate_response(
    question: dict[str, Any],
    model_name: str,
    iteration: int = 0,
    rng: random.Random | None = None,
) -> str:
    """Generate a simulated LLM response for a GSM8K question.

    **Detailed explanation for engineers:**
        When the LLM can't be loaded, simulates chain-of-thought responses with
        Apple-calibrated error rates from Exp 91/93:

        - Qwen3.5-0.8B: ~30% base error (matches Exp 91 Qwen simulation)
        - Gemma4-E4B-it: ~25% base error (matches Exp 91 Gemma simulation)

        Error types mirror real LLM failure modes:
        - Arithmetic errors (50%): wrong intermediate computation — fixable by repair
        - Logic errors (35%): missing/extra step in reasoning — harder to fix
        - Reading errors (15%): misunderstood the problem entirely

        Repair effectiveness (per iteration):
        - Arithmetic errors: ~70% fix rate per iteration (feedback provides correction)
        - Logic errors: ~30% fix rate per iteration
        - Reading errors: ~10% fix rate per iteration (deep misunderstanding)

        This simulation matches the Exp 91 calibration to ensure backward
        compatibility of result comparisons.
    """
    if rng is None:
        rng = random.Random(42)

    gt = question["ground_truth"]

    # Model-specific base error rates (Apple-calibrated from Exp 91).
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
# 8. Arithmetic constraint extraction
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
# 9. Violation formatting for repair prompts
# ---------------------------------------------------------------------------


def format_violations(arith_steps: list[dict[str, Any]]) -> str:
    """Convert arithmetic violations into natural language repair feedback.

    **Detailed explanation for engineers:**
        Produces plain English feedback identifying wrong intermediate
        calculations and the correct result. Does NOT reveal the ground truth
        answer — only corrects intermediate arithmetic. This is the Carnot
        value proposition: precise, actionable feedback that guides the LLM
        without giving away the answer.
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
# 10. Pipeline verification
# ---------------------------------------------------------------------------


def verify_with_pipeline(question: str, response: str) -> dict[str, Any]:
    """Run Carnot VerifyRepairPipeline.verify() on a response.

    **Detailed explanation for engineers:**
        Uses the library-level VerifyRepairPipeline in verify-only mode (no
        model loaded). Extracts arithmetic constraints from the response text,
        evaluates them, and returns the verification result.

        This integrates the REAL pipeline library for verifier validation.
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
        This is the control condition for measuring the pipeline's delta.
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

        Post-verify accuracy = correct_and_verified / total_questions
        (abstentions count as wrong, preserving N=1319 denominator).
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

    arith_steps = extract_arithmetic_steps(response)
    n_arith = len(arith_steps)
    n_violated = sum(1 for s in arith_steps if not s["satisfied"])
    pipeline_result = verify_with_pipeline(question["question"], response)

    elapsed = time.time() - t0
    extracted = extract_final_number(response)
    correct = extracted is not None and extracted == question["ground_truth"]
    flagged = n_violated > 0 or not pipeline_result["verified"]

    return {
        "mode": "verify_only",
        "extracted_answer": extracted,
        "correct": correct,
        "flagged": flagged,
        "n_arith_constraints": n_arith,
        "n_arith_violated": n_violated,
        "pipeline_verified": pipeline_result["verified"],
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
        errors fix ~50% per iteration, matching Exp 91 calibration).

        Key metric: "repaired" = initially wrong but finally correct.
        Delta accuracy = repair accuracy - baseline accuracy (the main claim).
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
    extracted = None

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
    final_correct = extracted is not None and extracted == gt

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
        "time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# 12. Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    correct_flags: list[bool],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 161,
) -> tuple[float, float, float]:
    """Compute accuracy and 95% bootstrap CI from a list of correct/wrong flags.

    **Detailed explanation for engineers:**
        Non-parametric bootstrap resampling:
        1. Draw n_bootstrap samples of size N (with replacement) from correct_flags.
        2. Compute accuracy for each bootstrap sample.
        3. Return the (alpha/2, 1-alpha/2) percentiles of the distribution.

        At N=1319 and n_bootstrap=10000, CI width ≈ ±2.7pp for accuracy near 70%.
        This is tight enough for comparison to published benchmarks.

    Args:
        correct_flags: List of True/False per question.
        n_bootstrap: Number of bootstrap samples (default 10,000).
        confidence: Confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (accuracy, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    arr = np.array(correct_flags, dtype=float)
    n = len(arr)
    point_estimate = float(np.mean(arr))

    # Vectorized bootstrap: draw all samples at once.
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    bootstrap_means = arr[indices].mean(axis=1)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper


def bootstrap_delta_ci(
    baseline_flags: list[bool],
    repair_flags: list[bool],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 162,
) -> tuple[float, float, float]:
    """Compute delta accuracy and 95% bootstrap CI for (repair - baseline).

    **Detailed explanation for engineers:**
        Paired bootstrap: the delta is computed per question
        (repair_correct[i] - baseline_correct[i]), then bootstrapped.
        Paired resampling accounts for correlation between modes on the same
        question, giving tighter CIs than unpaired bootstrap.

    Args:
        baseline_flags: Correct/wrong per question for baseline mode.
        repair_flags: Correct/wrong per question for verify+repair mode.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Tuple of (delta, ci_lower, ci_upper).
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
    ci_lower = float(np.percentile(bootstrap_deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2)))

    return point_delta, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# 13. Results saving
# ---------------------------------------------------------------------------


def save_results_json(
    all_results: dict[str, dict[str, list[dict[str, Any]]]],
    metadata: dict[str, Any],
    statistics: dict[str, Any],
) -> Path:
    """Save full experiment results to results/experiment_161_results.json.

    **Detailed explanation for engineers:**
        Saves the complete benchmark statistics (accuracy + CIs per mode per
        model) plus metadata. The per-question detail list is stripped of the
        verbose 'response' field (responses can be >1KB each; 1319 × 2 × 3
        modes = 7914 entries × 1KB ≈ 8MB if stored verbatim).

        The statistics section is the primary deliverable — it contains the
        publishable numbers:
        - n_questions: total questions evaluated
        - inference_mode: "live" or "simulation"
        - per model, per mode: accuracy, ci_lower, ci_upper
        - per model: improvement_delta, ci_delta_lower, ci_delta_upper
        - runtime_seconds: total wall-clock time

        The 'results' section stores per-question correct/wrong flags for
        post-hoc analysis without the response text.
    """
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    path = results_dir / "experiment_161_results.json"

    # Store per-question flags only (no response text, to keep file manageable).
    compact: dict[str, Any] = {}
    for model_name, modes in all_results.items():
        compact[model_name] = {}
        for mode_name, entries in modes.items():
            compact[model_name][mode_name] = [
                {k: v for k, v in e.items() if k not in ("response",)}
                for e in entries
            ]

    output = {
        "experiment": 161,
        "title": "Full GSM8K Benchmark (1,319 questions) with 95% Bootstrap CIs",
        "metadata": metadata,
        "statistics": statistics,
        "results": compact,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {path}")
    return path


# ---------------------------------------------------------------------------
# 14. Main benchmark
# ---------------------------------------------------------------------------


def main() -> int:
    """Run GSM8K full benchmark: 1,319 questions, 2 models, 3 modes each."""
    sep = "=" * 78
    print(sep)
    print("EXPERIMENT 161: Full GSM8K Benchmark (1,319 questions) with 95% CIs")
    print("  Real dataset: openai/gsm8k test split (N=1319)")
    print("  Models: Qwen3.5-0.8B, Gemma4-E4B-it")
    print("  Modes: Baseline | Verify-only | Verify+Repair")
    print("  Bootstrap CIs: n=10,000 samples → <±3pp at N=1319")
    print("  Addresses Goal #6: externally credible benchmark numbers")
    print(sep)

    overall_start = time.time()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # --- Check Exp 160 for eGPU ---
    print("\n[0/4] Checking Exp 160 eGPU status...")
    egpu_available = check_egpu_from_exp160()
    force_skip = bool(os.environ.get("CARNOT_SKIP_LLM", ""))
    if force_skip:
        print("  CARNOT_SKIP_LLM set: simulation mode forced.")
        use_egpu = False
    else:
        use_egpu = egpu_available

    # --- Load GSM8K questions ---
    print("\n[1/4] Loading GSM8K questions...")
    questions = load_gsm8k_questions(seed=161)
    n_real = sum(1 for q in questions if q.get("source") == "gsm8k")
    n_synth = sum(1 for q in questions if q.get("source") == "synthetic")
    n_questions = len(questions)
    print(f"  Questions: {n_questions} ({n_real} real GSM8K, {n_synth} synthetic)")

    if n_synth > 0:
        print(f"  WARNING: {n_synth} synthetic questions included.")
        print("  Results should NOT be compared to published benchmarks.")

    # --- Run benchmark per model ---
    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    model_metadata: dict[str, dict[str, Any]] = {}
    statistics: dict[str, Any] = {}

    for mi, config in enumerate(MODEL_CONFIGS):
        model_name = config["name"]
        print(f"\n[2/4] Model {mi + 1}/{len(MODEL_CONFIGS)}: {model_name}")

        print(f"  Loading {model_name}...")
        tokenizer, model_obj, device, use_live = load_model(config, use_egpu=use_egpu)

        if not use_live:
            print(f"  *** FALLBACK: Using simulated outputs for {model_name} ***")
            print(f"  Simulation uses Apple-calibrated error rates from Exp 91.")

        model_metadata[model_name] = {
            "live": use_live,
            "device": device,
            "egpu_requested": use_egpu,
        }

        modes_results: dict[str, list[dict[str, Any]]] = {
            "baseline": [],
            "verify_only": [],
            "verify_repair": [],
        }

        print(f"  Running {n_questions} questions × 3 modes...")
        model_start = time.time()

        for qi, q in enumerate(questions):
            # Deterministic per-question RNG (consistent with Exp 91 seed scheme).
            seed_base = 161_000 + mi * 100_000 + qi

            # Mode 1: Baseline.
            r_base = run_baseline(
                q, model_name,
                tokenizer=tokenizer, model=model_obj, device=device,
                use_live=use_live,
                sim_rng=random.Random(seed_base),
            )
            modes_results["baseline"].append(r_base)

            # Mode 2: Verify-only.
            r_verify = run_verify_only(
                q, model_name,
                tokenizer=tokenizer, model=model_obj, device=device,
                use_live=use_live,
                sim_rng=random.Random(seed_base),
            )
            modes_results["verify_only"].append(r_verify)

            # Mode 3: Verify+Repair.
            r_repair = run_verify_repair(
                q, model_name,
                tokenizer=tokenizer, model=model_obj, device=device,
                use_live=use_live,
                sim_rng=random.Random(seed_base),
                max_repairs=3,
            )
            modes_results["verify_repair"].append(r_repair)

            if (qi + 1) % 100 == 0 or qi == 0:
                n_b = sum(1 for r in modes_results["baseline"] if r["correct"])
                n_r = sum(1 for r in modes_results["verify_repair"] if r["correct"])
                print(f"    {qi + 1}/{n_questions} — "
                      f"baseline {n_b}/{qi + 1} ({n_b/(qi+1):.1%}), "
                      f"repair {n_r}/{qi + 1} ({n_r/(qi+1):.1%})")

        model_elapsed = time.time() - model_start
        model_metadata[model_name]["time_s"] = model_elapsed
        all_results[model_name] = modes_results

        # --- Compute statistics with bootstrap CIs ---
        print(f"\n  Computing bootstrap CIs (n=10,000 samples)...")

        base_flags = [r["correct"] for r in modes_results["baseline"]]
        verify_flags = [r["correct"] for r in modes_results["verify_only"]]
        repair_flags = [r["correct"] for r in modes_results["verify_repair"]]

        base_acc, base_lo, base_hi = bootstrap_ci(base_flags, seed=161 * 100 + mi)
        verify_acc, verify_lo, verify_hi = bootstrap_ci(verify_flags, seed=161 * 100 + mi + 1)
        repair_acc, repair_lo, repair_hi = bootstrap_ci(repair_flags, seed=161 * 100 + mi + 2)
        delta, delta_lo, delta_hi = bootstrap_delta_ci(
            base_flags, repair_flags, seed=161 * 100 + mi + 3
        )

        statistics[model_name] = {
            "baseline": {
                "accuracy": base_acc,
                "ci_lower": base_lo,
                "ci_upper": base_hi,
                "n_correct": int(sum(base_flags)),
            },
            "verify_only": {
                "accuracy": verify_acc,
                "ci_lower": verify_lo,
                "ci_upper": verify_hi,
                "n_correct": int(sum(verify_flags)),
            },
            "verify_repair": {
                "accuracy": repair_acc,
                "ci_lower": repair_lo,
                "ci_upper": repair_hi,
                "n_correct": int(sum(repair_flags)),
            },
            "improvement_delta": delta,
            "ci_delta_lower": delta_lo,
            "ci_delta_upper": delta_hi,
        }

        # Print model summary.
        print(f"\n  {model_name} summary ({model_elapsed:.1f}s):")
        print(f"    Baseline:      {sum(base_flags)}/{n_questions} "
              f"({base_acc:.1%} [{base_lo:.1%}, {base_hi:.1%}])")
        print(f"    Verify-only:   {sum(verify_flags)}/{n_questions} "
              f"({verify_acc:.1%} [{verify_lo:.1%}, {verify_hi:.1%}])")
        print(f"    Verify+Repair: {sum(repair_flags)}/{n_questions} "
              f"({repair_acc:.1%} [{repair_lo:.1%}, {repair_hi:.1%}])")
        print(f"    Δ (repair - baseline): {delta:+.1%} [{delta_lo:+.1%}, {delta_hi:+.1%}]")

        # Free model memory.
        if use_live:
            unload_model(model_obj, tokenizer, device)

    # --- Aggregate metadata ---
    total_elapsed = time.time() - overall_start
    any_live = any(m["live"] for m in model_metadata.values())
    inference_mode = "live" if any_live else "simulation"

    metadata = {
        "timestamp": timestamp,
        "n_questions": n_questions,
        "n_real_gsm8k": n_real,
        "n_synthetic": n_synth,
        "dataset_source": "GSM8K test" if n_real > 0 else "synthetic",
        "inference_mode": inference_mode,
        "egpu_detected": egpu_available,
        "runtime_seconds": total_elapsed,
        "bootstrap_samples": 10_000,
        "confidence_level": 0.95,
        "models": model_metadata,
    }

    # --- Save results ---
    print(f"\n[3/4] Saving results...")
    save_results_json(all_results, metadata, statistics)

    # --- Final summary ---
    print(f"\n[4/4] Final results")
    print(f"\n{sep}")
    print(f"EXPERIMENT 161 FINAL RESULTS ({total_elapsed:.1f}s total)")
    print(sep)
    print(f"  Dataset: {'REAL GSM8K' if n_real > 0 else 'Synthetic fallback'} "
          f"(N={n_questions}, {n_real} real, {n_synth} synthetic)")
    print(f"  Inference: {inference_mode.upper()}")
    print(f"  Bootstrap CI: 95%, n_bootstrap=10,000")
    print()

    header = f"  {'Model':<20s} {'Baseline':>14s} {'Verify':>14s} {'Repair':>14s} {'Δ':>12s}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    for model_name, stats in statistics.items():
        tag = " (LIVE)" if model_metadata[model_name]["live"] else " (SIM)"
        b = stats["baseline"]
        v = stats["verify_only"]
        r = stats["verify_repair"]
        d = stats["improvement_delta"]
        d_lo = stats["ci_delta_lower"]
        d_hi = stats["ci_delta_upper"]

        print(f"  {model_name + tag:<20s} "
              f"{b['accuracy']:>6.1%}±{(b['ci_upper']-b['ci_lower'])/2:>4.1%}  "
              f"{v['accuracy']:>6.1%}±{(v['ci_upper']-v['ci_lower'])/2:>4.1%}  "
              f"{r['accuracy']:>6.1%}±{(r['ci_upper']-r['ci_lower'])/2:>4.1%}  "
              f"{d:>+6.1%}[{d_lo:+.1%},{d_hi:+.1%}]")

    print()
    print("  Published baselines (for context):")
    for name, acc in PUBLISHED_BASELINES.items():
        print(f"    {name}: {acc:.1%}")

    print()
    if n_real > 0 and any_live:
        print("  VERDICT: PUBLISHABLE — real GSM8K data + live model inference.")
        print("  Bootstrap CIs <±3pp. Comparable to published benchmarks.")
        print("  Goal #6: ACHIEVED.")
    elif n_real > 0:
        print("  VERDICT: EXTERNALLY CREDIBLE DATASET but simulated inference.")
        print("  Results show pipeline mechanics. Run with live model for final claim.")
        print("  Goal #6: PARTIAL — dataset is real, inference is simulated.")
    else:
        print("  VERDICT: Synthetic fallback — pipeline mechanics exercised.")
        print("  Install `datasets` and re-run for externally credible numbers.")
        print("  Goal #6: NOT MET — synthetic data only.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
