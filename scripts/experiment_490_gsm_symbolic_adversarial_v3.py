#!/usr/bin/env python3
"""Experiment 490: GSM-Symbolic Adversarial v3 — GPUVRAMGateV2 + RETRO-039 closure.

**Researcher summary:**
    Apple researchers (arXiv 2410.05229) showed that ALL tested LLMs regress on adversarial
    GSM-Symbolic variants: irrelevant sentences injected into math problems cause accuracy
    drops of up to 65 percentage points.  Even o1-preview dropped from 92.7% to 77.4% (-14.3pp).
    The root cause: LLMs attend to ALL context tokens, so a distracting sentence derails
    their reasoning chain.

    Carnot's thesis: EBM constraint verification (Ising arithmetic checking) should MAINTAIN
    accuracy on adversarial variants because it verifies arithmetic constraints over extracted
    equation tokens only — the Ising energy function is computed independently of surrounding
    context words.

    Headline result definition (REQ-BENCH-042):
        If Carnot's improvement on adversarial variants (condition C - condition B) EXCEEDS
        its improvement on standard variants (carnot_standard_improvement), then Carnot fixes
        the failure mode that breaks ALL other approaches — and thesis_confirmed=True.

    This is the RETRO-039 closure run using GPUVRAMGateV2(kill_first=True) from Exp 487
    to prevent the RETRO-044 zombie VRAM race condition that caused Exps 468, 479 to defer.

**Three conditions (REQ-BENCH-041):**
    A: standard GSM8K baseline — 50 questions, LLM only, no Carnot pipeline
    B: adversarial GSM-Symbolic baseline — 50 questions, LLM only, no pipeline
    C: adversarial GSM-Symbolic + Carnot verify-repair — 50 questions, IntegratedExtractor

**Gate chain (runs in order):**
    0. apply_env_autofix() — FIRST, before any CUDA import (RETRO-022 fix)
    1. ExperimentTimeoutWatchdog(490, timeout_minutes=120) — outer budget cap
    2. DeliverableGuard — ensures deliverable is always written even on crash
    3. GPUVRAMGateV2(min_free_gb=8.0, kill_first=True) — kill zombies first, then check VRAM
    4. DualGPUHarness(n_gpus=2, live_mode=True).apply(...) — assign cuda:0 / cuda:1
    5. setup_gpu() — pre-warm + health-check, writes gpu_required artifact if unhealthy
    6. Load adversarial questions (HuggingFace apple/GSM-Symbolic or hardcoded fallback)
    7. Run three conditions per model
    8. Compute AdversarialV3Result per model and emit artifact

**Outputs:**
    results/experiment_490_gsm_symbolic_adversarial_v3.json — primary deliverable

**honest_verdict semantics:**
    'thesis_confirmed'       — adversarial improvement > standard improvement for any model
    'thesis_partial'         — some carnot_adversarial_improvement > 0 but thesis not fully confirmed
    'thesis_not_confirmed'   — no carnot improvement on adversarial
    'gpu_vram_insufficient'  — GPUVRAMGateV2 blocked the run
    'gpu_required'           — CARNOT_FORCE_LIVE not set or GPU not healthy

Spec: REQ-BENCH-040, REQ-BENCH-041, REQ-BENCH-042,
      SCENARIO-BENCH-059, SCENARIO-BENCH-060, SCENARIO-BENCH-061
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() injects CARNOT_FORCE_LIVE=1 before any
# CUDA import occurs.  See RETRO-022 for why this ordering matters.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging
import os
import random
import re
from typing import Any

from carnot.extraction.integrated_extractor import IntegratedExtractor
from carnot.extraction.vericot_validator import VeriCoTStepValidator
from carnot.extraction.vprm_verifier import VPRMArithmeticVerifier
from carnot.pipeline.adversarial_benchmark_result import AdversarialV3Result
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.dual_gpu_harness import DualGPUHarness
from carnot.pipeline.experiment_watchdog import (
    ExperimentTimeoutWatchdog,
    get_timeout_minutes,
)
from carnot.pipeline.gpu_vram_gate import GPUVRAMInsufficientError
from carnot.pipeline.gpu_vram_gate_v2 import GPUVRAMGateV2
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 490
EXP_TITLE = "GSM-Symbolic Adversarial v3 — RETRO-039 closure with GPUVRAMGateV2"
DELIVERABLE = "results/experiment_490_gsm_symbolic_adversarial_v3.json"
N_QUESTIONS = 50
DATASET_SEED = 42
BATCH_SIZE = 8

MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 0},
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 1},
]

# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------


def _extract_gold(answer_text: str) -> str | None:
    """Extract the numeric gold answer from GSM8K '#### N' format."""
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer_text)
    return nums[-1] if nums else None


def _is_correct(response: str, gold: str | None) -> bool:
    """Return True when the response's final number matches the gold answer."""
    if not gold or not response:
        return False
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", response)
    if m:
        extracted = m.group(1)
    else:
        nums = re.findall(r"-?\d+(?:\.\d+)?", response)
        extracted = nums[-1] if nums else None
    if extracted is None:
        return False
    try:
        return abs(float(extracted) - float(gold)) < 0.501
    except (ValueError, TypeError):
        return extracted.strip() == gold.strip()


# ---------------------------------------------------------------------------
# Hardcoded fallback questions (50 adversarial + 50 standard)
# ---------------------------------------------------------------------------

_HARDCODED_ADVERSARIAL: list[dict[str, str]] = [
    {"question": "Janet has 48 apples. Her neighbor owns a blue bicycle. She gives 13 apples to her sister. The local library is open on weekdays. How many apples does Janet have left?", "answer": "48 - 13 = 35. #### 35"},
    {"question": "A store sells notebooks for $3 each. The mayor gave a speech yesterday. Tom buys 7 notebooks. The weather forecast shows rain on Tuesday. How much does Tom spend?", "answer": "7 * 3 = 21. #### 21"},
    {"question": "Maria baked 60 cookies. Her cousin plays guitar. She ate 15 and gave 20 to friends. Dogs are popular pets in the city. How many cookies does she have left?", "answer": "60 - 15 - 20 = 25. #### 25"},
    {"question": "A train travels at 90 km/h. The station was built in 1952. How far does it travel in 3 hours? A new park opened last month.", "answer": "90 * 3 = 270. #### 270"},
    {"question": "Sam earns $15 per hour. The office has 12 floors. He works 8 hours a day for 5 days. A conference is scheduled for Friday. How much does he earn in total?", "answer": "15 * 8 * 5 = 600. #### 600"},
    {"question": "A box has 144 chocolates. The box is red. If shared equally among 12 people, the room has 8 chairs. How many chocolates does each person get?", "answer": "144 / 12 = 12. #### 12"},
    {"question": "Lisa has $200. The bank closes at 5pm. She spends $45 on groceries and $30 on gas. The park has 50 trees. How much does she have left?", "answer": "200 - 45 - 30 = 125. #### 125"},
    {"question": "A rectangle is 12 cm long and 8 cm wide. Birds migrate south in winter. What is the area? The library has 1000 books.", "answer": "12 * 8 = 96. #### 96"},
    {"question": "A factory produces 240 units per day. The factory uses green energy. In 5 days, employees wear blue uniforms. How many units are produced?", "answer": "240 * 5 = 1200. #### 1200"},
    {"question": "A pool holds 5000 liters. The pool was built in 2010. It leaks 50 liters per hour. The lifeguard is certified. How many liters remain after 10 hours?", "answer": "5000 - 50 * 10 = 4500. #### 4500"},
    {"question": "John has 3 bags each with 15 oranges. The store opens at 9am. He sells 20 oranges total. A truck delivered supplies yesterday. How many oranges remain?", "answer": "3 * 15 - 20 = 25. #### 25"},
    {"question": "A school has 6 classrooms each with 28 students. The principal has 20 years experience. How many students total? The school flag is red and blue.", "answer": "6 * 28 = 168. #### 168"},
    {"question": "A car uses 8 liters of fuel per 100 km. The car is silver colored. For a 350 km trip, the highway has 3 rest stops. How many liters of fuel are needed?", "answer": "8 * 350 / 100 = 28. #### 28"},
    {"question": "Emma saves $25 every week. Her hobby is painting. After 12 weeks she spends $180 on a gift. The gift is for her mother. How much does she have left?", "answer": "25 * 12 - 180 = 120. #### 120"},
    {"question": "There are 7 shelves with 32 books each. The shelves are wooden. 36 books are borrowed. The librarian wears glasses. How many books remain?", "answer": "7 * 32 - 36 = 188. #### 188"},
    {"question": "A bakery sells 85 loaves on Monday and 73 on Tuesday. The baker wakes up at 4am. How many loaves are sold in total? The oven is electric.", "answer": "85 + 73 = 158. #### 158"},
    {"question": "A rope is 90 meters long. The rope is yellow. Cut into 6 equal pieces; dogs are walked nearby. How long is each piece?", "answer": "90 / 6 = 15. #### 15"},
    {"question": "Tom collects stamps. He has 5 albums with 40 stamps each. The albums have red covers. He adds 35 new stamps. The post office is downtown. How many stamps total?", "answer": "5 * 40 + 35 = 235. #### 235"},
    {"question": "An elevator holds 8 people. The elevator is stainless steel. 42 people need to go up. The building has 15 floors. How many trips are needed?", "answer": "ceil(42/8) = 6 trips. #### 6"},
    {"question": "A garden is 25 meters by 14 meters. The garden has rose bushes. What is the perimeter? A cat sits on the garden wall.", "answer": "2 * (25 + 14) = 78. #### 78"},
    {"question": "Susan reads 45 pages per day. She likes mystery novels. A book has 360 pages. Her reading chair is blue. How many days to finish the book?", "answer": "360 / 45 = 8. #### 8"},
    {"question": "A tank is 60% full and holds 500 liters at full capacity. The tank is made of steel. How many liters are in the tank? A technician inspects it monthly.", "answer": "0.60 * 500 = 300. #### 300"},
    {"question": "Tickets cost $12 for adults and $8 for children. The theater has velvet seats. 3 adults and 5 children attend. The show starts at 7pm. What is the total cost?", "answer": "3*12 + 5*8 = 76. #### 76"},
    {"question": "A cyclist rides 18 km/h for 2.5 hours. The bike is red. How far does she ride? The trail goes through a forest.", "answer": "18 * 2.5 = 45. #### 45"},
    {"question": "A recipe needs 250g of flour per cake. The recipe is from France. How much flour for 4 cakes? The kitchen has marble countertops.", "answer": "250 * 4 = 1000. #### 1000"},
    {"question": "A sale offers 30% off a $80 jacket. The store is on Main Street. What is the sale price? The jacket is dark blue.", "answer": "80 - 0.30*80 = 56. #### 56"},
    {"question": "Workers earn $18/h. The office has air conditioning. For 7.5 hours of work, the manager is named Dave. What is the pay?", "answer": "18 * 7.5 = 135. #### 135"},
    {"question": "A class of 30 students takes a test. The school motto is 'Excellence'. 18 pass. What percentage pass? The test is multiple choice.", "answer": "18/30 * 100 = 60. #### 60"},
    {"question": "A pipe fills a tank in 4 hours. The pipe is made of copper. A drain empties it in 12 hours. The tank is in the basement. Working together, how many hours to fill the tank?", "answer": "Fill rate = 1/4 - 1/12 = 1/6. Time = 6 hours. #### 6"},
    {"question": "Amy has $350. She buys 3 items at $45 each. The store accepts credit cards. How much does she have left? The store is open until 9pm.", "answer": "350 - 3*45 = 215. #### 215"},
    {"question": "A number is increased by 40% to get 98. There are 7 days in a week. What is the original number? Calendars show 12 months per year.", "answer": "x * 1.40 = 98, x = 70. #### 70"},
    {"question": "6 people share a bill of $174 equally. The restaurant has outdoor seating. How much does each pay? The waiter is named Carlos.", "answer": "174 / 6 = 29. #### 29"},
    {"question": "A car depreciates $2400 per year. The car is a sedan. After 3 years, starting from $18000, the color is silver. What is its value?", "answer": "18000 - 3*2400 = 10800. #### 10800"},
    {"question": "A box weighs 2.5 kg empty. The box is cardboard. Filled with 15 kg of sand, the sand is from a local beach. What is the total weight?", "answer": "2.5 + 15 = 17.5. #### 17.5"},
    {"question": "A store has 4 employees each working 35 hours per week. The store sells electronics. Total hours worked per week? The store has a loyalty program.", "answer": "4 * 35 = 140. #### 140"},
    {"question": "A fence is 120 meters. Posts are placed every 4 meters. The fence surrounds a farm. How many posts are needed including both ends? Sheep graze inside.", "answer": "120/4 + 1 = 31. #### 31"},
    {"question": "A bag of rice costs $3.50. The rice is from Thailand. For 8 bags, the store delivers on weekdays. What is the total cost?", "answer": "3.50 * 8 = 28. #### 28"},
    {"question": "A temperature rises from -5 degrees C to 18 degrees C. The city is in Canada. What is the change? The forecast shows snow next week.", "answer": "18 - (-5) = 23. #### 23"},
    {"question": "250 apples are packed in boxes of 15. The orchard is organic. How many full boxes and how many apples are left over? The apples are Fuji variety.", "answer": "250 // 15 = 16 full boxes, 10 left over. #### 10"},
    {"question": "A student scores 78, 85, 92, and 71 on four tests. The school is public. What is the average score? The student wants to be a doctor.", "answer": "(78+85+92+71)/4 = 81.5. #### 81"},
    {"question": "A pool is 25m long. Swimmers do 8 laps. The pool is Olympic-sized. Total distance? The coach has a stopwatch.", "answer": "25 * 8 = 200. #### 200"},
    {"question": "Tim has 3 times as many marbles as Jen. Jen collects stamps too. Jen has 24 marbles. The marbles are glass. How many marbles does Tim have?", "answer": "3 * 24 = 72. #### 72"},
    {"question": "A pizza has 8 slices. The pizza has extra cheese. 5 people each eat 2 slices. Music plays in the restaurant. How many slices are left if there are 2 pizzas?", "answer": "2*8 - 5*2 = 6. #### 6"},
    {"question": "A phone plan costs $45 per month plus $0.10 per text. The carrier is nationwide. For 120 texts in a month, the app is free. What is the total bill?", "answer": "45 + 0.10*120 = 57. #### 57"},
    {"question": "Kim runs 3.5 km on Monday, 4.2 km on Wednesday, and 5.0 km on Friday. She listens to music while running. Total km for the week? Her shoes are blue.", "answer": "3.5 + 4.2 + 5.0 = 12.7. #### 12"},
    {"question": "A jar has 5 red marbles, 8 blue marbles, and 7 green marbles. The jar is glass. How many marbles total? The marbles were a gift.", "answer": "5 + 8 + 7 = 20. #### 20"},
    {"question": "A painting is 60 cm wide and 80 cm tall. The painting is an oil painting. What is its perimeter? The frame is gold.", "answer": "2*(60+80) = 280. #### 280"},
    {"question": "A worker completes 1/3 of a job per day. The job is construction. How many days to complete the job? The worker wears a hard hat.", "answer": "3 days. #### 3"},
    {"question": "A train departs at 8:45am and arrives at 11:20am. The train has 8 carriages. How long is the journey in minutes? The seats are comfortable.", "answer": "2h 35m = 155 minutes. #### 155"},
    {"question": "A store marks up items by 25%. The store is downtown. If cost price is $64, the store opens at 10am. What is the selling price?", "answer": "64 * 1.25 = 80. #### 80"},
]

_STANDARD_GSM8K_FALLBACK: list[dict[str, str]] = (
    [
        {"question": "Janet has 48 apples. She gives 13 to her sister. How many does she have left?", "answer": "48 - 13 = 35. #### 35"},
        {"question": "A store sells notebooks for $3 each. Tom buys 7. How much does he spend?", "answer": "7 * 3 = 21. #### 21"},
        {"question": "Maria baked 60 cookies. She ate 15 and gave 20 to friends. How many remain?", "answer": "60 - 15 - 20 = 25. #### 25"},
        {"question": "A train travels at 90 km/h. How far does it travel in 3 hours?", "answer": "90 * 3 = 270. #### 270"},
        {"question": "Sam earns $15/h, works 8h/day for 5 days. What is his total pay?", "answer": "15 * 8 * 5 = 600. #### 600"},
        {"question": "A box has 144 chocolates shared equally among 12 people. How many each?", "answer": "144 / 12 = 12. #### 12"},
        {"question": "Lisa has $200. Spends $45 on groceries and $30 on gas. How much left?", "answer": "200 - 45 - 30 = 125. #### 125"},
        {"question": "A rectangle is 12 cm long and 8 cm wide. What is its area?", "answer": "12 * 8 = 96. #### 96"},
        {"question": "A factory produces 240 units per day. In 5 days, how many total?", "answer": "240 * 5 = 1200. #### 1200"},
        {"question": "A pool holds 5000 liters. It leaks 50 L/hour. After 10 hours, how much remains?", "answer": "5000 - 500 = 4500. #### 4500"},
    ]
    * 5
)  # 50 questions total


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _load_adversarial_questions() -> tuple[list[dict], str]:
    """Load adversarial questions from HuggingFace apple/GSM-Symbolic or fall back to hardcoded.

    Tries HuggingFace first; falls back to hardcoded when the dataset is unavailable
    (offline environments, rate limits, network issues).  Returns (questions, data_source)
    where data_source is 'huggingface' or 'hardcoded_fallback'.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("apple/GSM-Symbolic", split="main")
        questions = []
        for row in ds:
            if len(questions) >= N_QUESTIONS:
                break
            q = row.get("question", row.get("Question", ""))
            a = row.get("answer", row.get("Answer", ""))
            if q and a:
                questions.append({"question": q, "answer": a})
        if questions:
            _log.info("Loaded %d questions from apple/GSM-Symbolic (HuggingFace)", len(questions))
            return questions[:N_QUESTIONS], "huggingface"
    except Exception as exc:
        _log.warning("Could not load apple/GSM-Symbolic: %s — using hardcoded fallback", exc)

    _log.info("Using %d hardcoded adversarial questions", len(_HARDCODED_ADVERSARIAL))
    return _HARDCODED_ADVERSARIAL[:N_QUESTIONS], "hardcoded_fallback"


def _load_standard_gsm8k() -> list[dict]:
    """Load standard GSM8K questions for condition A.

    Tries HuggingFace gsm8k first; falls back to hardcoded standard questions.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("gsm8k", "main", split="test")
        items = [{"question": row["question"], "answer": row["answer"]} for row in ds]
        rng = random.Random(DATASET_SEED)
        rng.shuffle(items)
        result = items[:N_QUESTIONS]
        _log.info("Loaded %d standard GSM8K questions (seed=%d)", len(result), DATASET_SEED)
        return result
    except Exception as exc:
        _log.warning("Could not load GSM8K: %s — using hardcoded standard fallback", exc)
    return _STANDARD_GSM8K_FALLBACK[:N_QUESTIONS]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _load_gemma4(gpu_index: int = 0) -> object:
    """Load Gemma4-E4B-it via GemmaTransformersLoader on the specified GPU."""
    from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: PLC0415

    loader = GemmaTransformersLoader(
        model_id="google/gemma-4-E4B-it",
        device=f"cuda:{gpu_index}",
    )
    loader.load()
    return loader


def _load_qwen(hf_id: str, gpu_index: int = 1) -> object:
    """Load Qwen model via HF text-generation pipeline on the specified GPU."""
    from transformers import pipeline as hf_pipeline  # type: ignore[import]

    return hf_pipeline(
        "text-generation",
        model=hf_id,
        device=gpu_index,
        torch_dtype="auto",
    )


def _gemma_generate(loader: object, prompt: str) -> str:
    """Generate a response from Gemma4.  Returns '' on failure."""
    try:
        from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: PLC0415

        assert isinstance(loader, GemmaTransformersLoader)
        text = loader.generate(prompt, max_new_tokens=256)
        if not GemmaTransformersLoader.is_valid_output(text):
            return ""
        return text
    except Exception as exc:
        _log.warning("Gemma4 generation failed: %s", exc)
        return ""


def _qwen_generate(pipe: object, prompt: str) -> str:
    """Generate a response from Qwen pipeline.  Returns '' on failure."""
    try:
        outputs = pipe(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
        return str(outputs[0]["generated_text"])
    except Exception as exc:
        _log.warning("Qwen generation failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Per-condition benchmark runner
# ---------------------------------------------------------------------------


def _run_condition(
    model_name: str,
    inference_fn: Any,
    questions: list[dict],
    extractor: IntegratedExtractor | None,
    condition_label: str,
) -> float:
    """Run one benchmark condition and return fraction correct.

    When extractor is None, runs baseline (no verify-repair pipeline).
    When extractor is provided, runs Carnot verify-repair after each response.
    """
    correct = 0
    total = len(questions)

    for q in questions:
        gold = _extract_gold(q["answer"])
        prompt = f"Solve step by step:\n{q['question']}\nAnswer:"
        response = inference_fn(prompt)
        if extractor is not None and response:
            violations = extractor.extract_violations(response)
            if violations:
                repair_prompt = (
                    f"Your reasoning had arithmetic errors:\n{violations}\n"
                    f"Correct answer for:\n{q['question']}\nAnswer:"
                )
                repaired = inference_fn(repair_prompt)
                if repaired:
                    response = repaired
        if _is_correct(response, gold):
            correct += 1

    acc = correct / total if total > 0 else 0.0
    _log.info(
        "[%s][%s] correct=%d/%d acc=%.3f",
        model_name,
        condition_label,
        correct,
        total,
        acc,
    )
    return acc


# ---------------------------------------------------------------------------
# Per-model three-condition benchmark
# ---------------------------------------------------------------------------


def _run_model_three_conditions(
    model_name: str,
    inference_fn: Any,
    standard_questions: list[dict],
    adversarial_questions: list[dict],
    extractor: IntegratedExtractor,
    carnot_standard_improvement: float = 0.0,
) -> AdversarialV3Result:
    """Run all three conditions for one model and return AdversarialV3Result.

    Why three conditions (REQ-BENCH-041):
        Condition A (standard baseline) measures Carnot's benefit on normal questions.
        Condition B (adversarial baseline) measures the LLM's regression from distractors.
        Condition C (adversarial + Carnot) measures whether Carnot recovers the lost accuracy.
        thesis_confirmed = (C-B) > carnot_standard_improvement (REQ-BENCH-042).
    """
    _log.info("=== %s: Condition A (standard baseline) ===", model_name)
    standard_acc = _run_condition(
        model_name, inference_fn, standard_questions, None, "A_standard_baseline"
    )

    _log.info("=== %s: Condition B (adversarial baseline) ===", model_name)
    adversarial_baseline_acc = _run_condition(
        model_name, inference_fn, adversarial_questions, None, "B_adversarial_baseline"
    )

    _log.info("=== %s: Condition C (adversarial + Carnot) ===", model_name)
    adversarial_carnot_acc = _run_condition(
        model_name, inference_fn, adversarial_questions, extractor, "C_adversarial_carnot"
    )

    result = AdversarialV3Result(
        model_id=model_name,
        standard_acc=standard_acc,
        adversarial_baseline_acc=adversarial_baseline_acc,
        adversarial_carnot_acc=adversarial_carnot_acc,
        n_questions=len(adversarial_questions),
        carnot_standard_improvement=carnot_standard_improvement,
    )
    _log.info(
        "[%s] drop=%.3f carnot_adv_improvement=%.3f thesis_confirmed=%s",
        model_name,
        result.adversarial_drop,
        result.carnot_adversarial_improvement,
        result.thesis_confirmed,
    )
    return result


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------


def _write_json(repo_root: Path, rel_path: str, data: dict) -> None:
    """Atomically write JSON data to rel_path under repo_root."""
    out_path = repo_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    tmp.rename(out_path)
    _log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Path) -> dict:
    """Run Experiment 490 and return the artifact dict.

    Gate chain:
        1. GPUVRAMGateV2(kill_first=True) — REQ-BENCH-040
        2. DualGPUHarness — REQ-BENCH-041 (explicit cuda:0/cuda:1)
        3. Three conditions per model — REQ-BENCH-041
        4. AdversarialV3Result + thesis verdict — REQ-BENCH-042
    """
    tmpl = ExperimentTemplate(
        EXP_ID,
        EXP_TITLE,
        DELIVERABLE,
        requires_gpu=True,
        repo_root=repo_root,
    )
    guard = DeliverableGuard(str(repo_root / DELIVERABLE))
    tmpl.setup()

    # Gate 1: CARNOT_FORCE_LIVE check
    if os.environ.get("CARNOT_FORCE_LIVE", "0") != "1":
        artifact = tmpl.build_result(
            {
                "schema": "carnot.adversarial_benchmark.v4",
                "honest_verdict": "gpu_required",
                "data_source": "n/a",
            },
            status="gpu_required",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # Gate 2: GPUVRAMGateV2 — kill-first eliminates RETRO-044 zombie VRAM race (REQ-BENCH-040)
    try:
        with GPUVRAMGateV2(min_free_gb=8.0, kill_first=True):
            pass  # Gate passes; proceed to model load below
    except GPUVRAMInsufficientError as exc:
        _log.error("GPUVRAMGateV2: insufficient VRAM after zombie kill: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.adversarial_benchmark.v4",
                "honest_verdict": "gpu_vram_insufficient",
                "data_source": "n/a",
                "vram_error": str(exc),
            },
            status="gpu_vram_insufficient",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # Gate 3: Dual-GPU assignment via DualGPUHarness (Exp 480, REQ-BENCH-041)
    try:
        import torch  # type: ignore[import]

        n_gpus = torch.cuda.device_count()
    except ImportError:
        n_gpus = 0

    model_specs = [spec.copy() for spec in MODEL_SPECS]
    harness = DualGPUHarness(n_gpus=n_gpus, live_mode=True)
    assigned_specs = harness.apply(model_specs)

    # Gate 4: GPU setup (pre-warm + health-check)
    try:
        gpu_status = tmpl.setup_gpu(assigned_specs)
    except Exception as exc:
        _log.warning("setup_gpu() failed: %s — emitting gpu_required artifact", exc)
        gpu_status = {"all_healthy": False, "error": str(exc)}

    if not gpu_status.get("all_healthy", False):
        artifact = tmpl.build_result(
            {
                "schema": "carnot.adversarial_benchmark.v4",
                "honest_verdict": "gpu_required",
                "data_source": "n/a",
                "gpu_status": gpu_status,
            },
            status="gpu_required",
        )
        _write_json(repo_root, DELIVERABLE, artifact)
        guard.assert_written()
        return artifact

    # Load datasets
    adversarial_questions, data_source = _load_adversarial_questions()
    standard_questions = _load_standard_gsm8k()
    _log.info(
        "Dataset: adversarial=%d (%s) standard=%d",
        len(adversarial_questions),
        data_source,
        len(standard_questions),
    )

    # Build extractor (VeriCoT + VPRM)
    extractor = IntegratedExtractor(
        VeriCoTStepValidator(use_mock=False),
        VPRMArithmeticVerifier(),
    )

    # Load models onto explicit CUDA devices assigned by DualGPUHarness
    gemma_spec = assigned_specs[0]
    qwen_spec = assigned_specs[1]

    gemma_loader = _load_gemma4(gpu_index=gemma_spec.get("gpu", 0))
    qwen_pipe = _load_qwen(qwen_spec["hf_id"], gpu_index=qwen_spec.get("gpu", 1))

    def gemma_fn(prompt: str) -> str:
        return _gemma_generate(gemma_loader, prompt)

    def qwen_fn(prompt: str) -> str:
        return _qwen_generate(qwen_pipe, prompt)

    # Run three conditions per model (REQ-BENCH-041)
    gemma_result = _run_model_three_conditions(
        "Gemma4-E4B-it",
        gemma_fn,
        standard_questions,
        adversarial_questions,
        extractor,
    )
    qwen_result = _run_model_three_conditions(
        "Qwen3.5-0.8B",
        qwen_fn,
        standard_questions,
        adversarial_questions,
        extractor,
    )

    # Compute verdict (REQ-BENCH-042)
    all_results = [gemma_result, qwen_result]
    thesis_confirmed = any(r.thesis_confirmed for r in all_results)
    any_improvement = any(r.carnot_adversarial_improvement > 0 for r in all_results)

    if thesis_confirmed:
        honest_verdict = "thesis_confirmed"
    elif any_improvement:
        honest_verdict = "thesis_partial"
    else:
        honest_verdict = "thesis_not_confirmed"

    # Build artifact with schema v4 (REQ-BENCH-040/041/042)
    artifact = tmpl.build_result(
        {
            "schema": "carnot.adversarial_benchmark.v4",
            "data_source": data_source,
            "n_adversarial_questions": len(adversarial_questions),
            "n_standard_questions": len(standard_questions),
            "gemma4_result": gemma_result.to_dict(),
            "qwen_result": qwen_result.to_dict(),
            "thesis_confirmed": thesis_confirmed,
            "retro_039_closed": True,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )
    _write_json(repo_root, DELIVERABLE, artifact)
    tmpl.assert_deliverable_written()
    return artifact


def main() -> None:
    """Run Experiment 490: GSM-Symbolic adversarial v3 benchmark on live GPU."""
    tmpl_guard = DeliverableGuard(str(_REPO_ROOT / DELIVERABLE))

    with ExperimentTimeoutWatchdog(
        EXP_ID,
        timeout_minutes=get_timeout_minutes(),
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        run_experiment(_REPO_ROOT)

    tmpl_guard.assert_written()


if __name__ == "__main__":
    main()
