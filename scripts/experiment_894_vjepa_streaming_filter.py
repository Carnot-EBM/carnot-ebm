#!/usr/bin/env python3
"""Experiment 894: VJEPA Streaming Logit Filter — generation-time constraint guidance.

**Why this experiment:**
    VJEPA v2 (Exp 884, ood_auc=0.9211) currently acts as a *post-hoc* Tier 2 filter
    that rejects completed CoT traces.  Post-hoc rejection is expensive: an entire
    generation pass completes before the violation is caught.  This experiment tests
    whether wiring VJEPA as a HuggingFace LogitsProcessor reduces violations DURING
    generation — moving the enforcement boundary from output to token-selection time.

    Connection to arXiv 2502.03685 (Discrete Autoregressive Biasing) and
    arXiv 2603.03305 (Draft-Conditioned Constrained Decoding): both papers show
    soft logit penalties at each step can reduce constraint violations without
    requiring rejection sampling.

**What we measure:**
    - baseline accuracy on 25 GSM8K questions (Gemma4-E4B-it, no streaming filter)
    - streaming accuracy with VJEPAStreamingLogitsProcessor enabled
    - signed_improvement = streaming_correct - baseline_correct
    - streaming_filter_applied_count: how many generation steps VJEPA triggered
    - streaming_filter_applied_pct: fraction of steps where penalty fired

**Gate condition:**
    Requires CARNOT_FORCE_LIVE=1 and GPU access (ROCm group via sg render).
    Without GPU, produces honest_verdict="streaming_blocked_no_gpu".

**honest_verdict:**
    - "streaming_positive"   if signed_improvement > 0
    - "streaming_neutral"    if signed_improvement == 0
    - "streaming_negative"   if signed_improvement < 0
    - "streaming_blocked_no_gpu"  if GPU not available

Spec: REQ-VERIFY-177, SCENARIO-VERIFY-177
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from experiment_template import ExperimentTemplate

RESULT_PATH = _ROOT / "results" / "experiment_894_vjepa_streaming_filter.json"
EXP_884_RESULT_PATH = _ROOT / "results" / "experiment_884_vjepa_cascade_deploy.json"

# 25 representative GSM8K questions with their numeric answers.
# These are standard grade-school math problems drawn from the GSM8K test split.
# We embed them directly to avoid a network dependency (the full test split is
# available at HuggingFace datasets, but downloading it during an experiment
# adds latency and a potential failure mode).
GSM8K_SAMPLE: list[dict[str, Any]] = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": 72,
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": 10,
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": 5,
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read tomorrow?",
        "answer": 42,
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": 624,
    },
    {
        "question": "Mark has a garden with flowers. He planted plants of five different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have?",
        "answer": 35,
    },
    {
        "question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
        "answer": 48,
    },
    {
        "question": "Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he piled on 2 pounds of jelly beans and 5 pounds of chocolate bars.  Then he added 2 pounds of gummy worms.  He noticed the scale read 15 pounds.  How many pounds do the box's contents weigh?",
        "answer": 9,
    },
    {
        "question": "Alexis is applying for a new job and bought a new set of clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a blouse, $46 on a skirt, $38 on a pair of shoes, and $11 on accessories. How much money does Alexis have left?",
        "answer": 75,
    },
    {
        "question": "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?",
        "answer": 990,
    },
    {
        "question": "A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?",
        "answer": 121,
    },
    {
        "question": "Tobias is buying a new pair of shoes that costs $95. He has been saving up his allowance for several weeks. He gets a $5 allowance per week. He has also been doing extra chores, which he gets paid $20 for each time he does them. He has done extra chores 3 times. How many weeks has he been saving his allowance if he just enough money to buy the shoes?",
        "answer": 7,
    },
    {
        "question": "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all?",
        "answer": 85,
    },
    {
        "question": "Jasmine wants to organize her bookcases. She has 4 bookcases, each of which can have 4 shelves. She currently has 2 shelves in each bookcase. If she wants to add 2 more shelves to each bookcase, how many more shelves does she need?",
        "answer": 8,
    },
    {
        "question": "A baker makes chocolate muffins and peanut butter muffins. His recipes make 10 muffins each. He makes 4 chocolate muffin recipes and 5 peanut butter muffin recipes. If 10 of the muffins are accidentally burned, how many muffins does the baker have left?",
        "answer": 80,
    },
    {
        "question": "A restaurant makes 40 more pizzas than hotdogs every day. If the restaurant makes 60 hotdogs every day, how many pizzas and hotdogs does it make every week?",
        "answer": 700,
    },
    {
        "question": "Wendy's truck has a gas tank that can hold 20 gallons. She also has a car with a gas tank that holds 12 gallons. The truck's tank is 3/4 full. The car's tank is 1/2 full. If she fills them both up completely, how many gallons does she add?",
        "answer": 11,
    },
    {
        "question": "Adam needs a new laptop and has two choices. The first laptop is $500. The second laptop is 3 times as expensive as the first laptop. How much would Adam have to spend if he decides to buy both?",
        "answer": 2000,
    },
    {
        "question": "There are 25 roses in a garden. There are 40 tulips. There are 35 daisies. What percentage of flowers are not roses?",
        "answer": 75,
    },
    {
        "question": "Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. How many filed does he have on his computer now?",
        "answer": 240,
    },
    {
        "question": "A store sells pencils, pens and markers. A pencil costs $0.5, a pen costs $1 and a marker costs $2. Ana bought 20 pencils, 10 pens, and 5 markers. How much did Ana spend?",
        "answer": 30,
    },
    {
        "question": "The gauge on a water tank shows that the tank is 1/3 full of water. To fill the tank, 16 gallons of water are added. How many gallons of water does the tank hold when full?",
        "answer": 24,
    },
    {
        "question": "Lucy lost 3 kg in the first week of her diet. She lost twice that many the second week, and the third week she lost half of what she lost the second week. How many kg has she lost total in three weeks?",
        "answer": 12,
    },
    {
        "question": "Brinley's teacher said that she will increase the time allowed for the next test by 25 minutes. If the time for the last test was 45 minutes, how long is the time allowed for the next test?",
        "answer": 70,
    },
    {
        "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "answer": 540,
    },
]


def _extract_numeric_answer(text: str) -> float | None:
    """Extract the final numeric answer from a generated text.

    Looks for patterns like '#### 42', 'The answer is 42', or the last number
    in the text.  Returns None if no number can be extracted.

    We intentionally keep this simple: the goal is to detect whether VJEPA
    steering changes the *direction* of correctness, not to build a perfect
    answer extractor.
    """
    # Try '#### N' pattern (GSM8K standard format)
    m = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1).replace(",", ""))

    # Try 'the answer is N' variants
    m = re.search(r"answer\s+is\s+([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))

    # Last number in the text
    nums = re.findall(r"([\d,]+(?:\.\d+)?)", text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            return None
    return None


def _is_correct(generated: str, expected: int | float) -> bool:
    """Return True if the generated answer matches the expected numeric answer."""
    pred = _extract_numeric_answer(generated)
    if pred is None:
        return False
    return abs(pred - expected) < 0.5


def load_vjepa_model(model_path: str) -> Any:
    """Load VJEPA v2 weights from safetensors.

    Returns a VariationalJEPAPredictor with loaded parameters, or None if the
    safetensors file does not exist (which prevents the experiment from running
    in environments where Exp 884 has not been executed).

    Args:
        model_path: Absolute path to the .safetensors file.
    """
    import jax.numpy as jnp

    from python.carnot.models.vjepa_predictor import (
        VOCAB_SIZE,
        VariationalJEPAPredictor,
    )

    p = Path(model_path)
    if not p.exists():
        return None

    try:
        from safetensors import safe_open

        model = VariationalJEPAPredictor(in_dim=VOCAB_SIZE, context_dim=VOCAB_SIZE)
        tensors: dict[str, Any] = {}
        with safe_open(str(p), framework="numpy") as f:
            for key in f.keys():
                tensors[key] = jnp.array(f.get_tensor(key))
        model.set_all_params(tensors)
        return model
    except Exception:
        return None


def assign_honest_verdict(signed_improvement: int, gpu_available: bool) -> str:
    """Map measured improvement to the honest_verdict string.

    Args:
        signed_improvement: streaming_correct - baseline_correct (integer).
        gpu_available:       Whether GPU was accessible during generation.

    Returns:
        One of: streaming_positive, streaming_neutral, streaming_negative,
        streaming_blocked_no_gpu.
    """
    if not gpu_available:
        return "streaming_blocked_no_gpu"
    if signed_improvement > 0:
        return "streaming_positive"
    if signed_improvement == 0:
        return "streaming_neutral"
    return "streaming_negative"


def _run_generation_baseline(
    model: Any, tokenizer: Any, questions: list[dict[str, Any]]
) -> tuple[int, float]:
    """Run baseline generation (no streaming filter) on all questions.

    Args:
        model:      Loaded HuggingFace model.
        tokenizer:  Matching tokenizer.
        questions:  List of {question, answer} dicts.

    Returns:
        (correct_count, avg_tokens_per_question)
    """
    import torch

    correct = 0
    total_tokens = 0

    for item in questions:
        prompt = f"Solve the following math problem step by step.\n\nQuestion: {item['question']}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        n_new = output.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += n_new
        generated_text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        if _is_correct(generated_text, item["answer"]):
            correct += 1

    avg_tokens = total_tokens / max(len(questions), 1)
    return correct, avg_tokens


def _run_generation_streaming(
    model: Any,
    tokenizer: Any,
    questions: list[dict[str, Any]],
    processor: Any,
) -> tuple[int, int]:
    """Run generation with VJEPAStreamingLogitsProcessor on all questions.

    Args:
        model:      Loaded HuggingFace model.
        tokenizer:  Matching tokenizer.
        questions:  List of {question, answer} dicts.
        processor:  VJEPAStreamingLogitsProcessor instance.

    Returns:
        (correct_count, total_applied_count)
    """
    import torch

    correct = 0

    for item in questions:
        prompt = f"Solve the following math problem step by step.\n\nQuestion: {item['question']}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                logits_processor=[processor],
            )
        generated_text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        if _is_correct(generated_text, item["answer"]):
            correct += 1

    return correct, processor.applied_count


def _write_blocked_artifact(reason: str, tmpl: ExperimentTemplate) -> None:
    """Write a blocked artifact and exit."""
    artifact = tmpl.build_result(
        {"blocked_reason": reason},
        status="blocked",
        honest_verdict="streaming_blocked_no_gpu",
        vjepa_model_path=str(EXP_884_RESULT_PATH),
        vjepa_ood_auc=0.9211,
    )
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w") as fh:
        json.dump(artifact, fh, indent=2)
    tmpl.assert_deliverable_written()
    sys.exit(0)


def main() -> None:
    """Run Exp 894: VJEPA Streaming Logit Filter."""
    tmpl = ExperimentTemplate(
        exp_id=894,
        title="VJEPA Streaming Logit Filter — generation-time constraint guidance",
        deliverable=str(RESULT_PATH),
        requires_gpu=True,
    )
    tmpl.setup()

    force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"

    # Read Exp 884 artifact for model path
    if not EXP_884_RESULT_PATH.exists():
        _write_blocked_artifact("exp884 artifact not found", tmpl)
        return

    with EXP_884_RESULT_PATH.open() as fh:
        exp884 = json.load(fh)
    vjepa_model_path = exp884.get("model_path", "")
    vjepa_ood_auc = exp884.get("final_ood_auc", 0.0)

    if not force_live:
        # Dry-run mode: produce a synthetic result for CI
        artifact = tmpl.build_result(
            {
                "gsm8k_accuracy_baseline": 0.0,
                "gsm8k_accuracy_streaming": 0.0,
                "signed_improvement": 0,
                "streaming_filter_applied_count": 0,
                "streaming_filter_applied_pct": 0.0,
                "violation_threshold": 0.75,
                "avg_tokens_per_question": 0.0,
                "vjepa_model_path": vjepa_model_path,
                "vjepa_ood_auc": vjepa_ood_auc,
                "mode": "dry_run",
            },
            status="blocked",
            honest_verdict="streaming_blocked_no_gpu",
        )
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RESULT_PATH.open("w") as fh:
            json.dump(artifact, fh, indent=2)
        tmpl.assert_deliverable_written()
        return

    # Load VJEPA model
    vjepa = load_vjepa_model(vjepa_model_path)
    if vjepa is None:
        _write_blocked_artifact(f"VJEPA safetensors not found at {vjepa_model_path}", tmpl)
        return

    # Load Gemma4-E4B-it via transformers
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
        gen_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-4-E4B-it",
            device_map="auto",
        )
    except Exception as exc:
        _write_blocked_artifact(f"model load failed: {exc}", tmpl)
        return

    from python.carnot.pipeline.vjepa_streaming_processor import VJEPAStreamingLogitsProcessor

    questions = GSM8K_SAMPLE[:25]

    # Baseline run
    baseline_correct, avg_tokens = _run_generation_baseline(gen_model, tokenizer, questions)

    # Streaming run
    processor = VJEPAStreamingLogitsProcessor(vjepa, tokenizer)
    streaming_correct, applied_count = _run_generation_streaming(
        gen_model, tokenizer, questions, processor
    )

    signed_improvement = streaming_correct - baseline_correct
    total_steps = max(applied_count + (25 * int(avg_tokens)), 1)
    applied_pct = applied_count / total_steps

    verdict = assign_honest_verdict(signed_improvement, gpu_available=True)

    artifact = tmpl.build_result(
        {
            "gsm8k_accuracy_baseline": baseline_correct / 25,
            "gsm8k_accuracy_streaming": streaming_correct / 25,
            "signed_improvement": signed_improvement,
            "streaming_filter_applied_count": applied_count,
            "streaming_filter_applied_pct": applied_pct,
            "violation_threshold": 0.75,
            "avg_tokens_per_question": avg_tokens,
            "vjepa_model_path": vjepa_model_path,
            "vjepa_ood_auc": vjepa_ood_auc,
            "mode": "live",
        },
        status="success",
        honest_verdict=verdict,
    )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w") as fh:
        json.dump(artifact, fh, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
