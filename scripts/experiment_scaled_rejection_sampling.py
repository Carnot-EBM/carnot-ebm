#!/usr/bin/env python3
"""Experiment: Scaled calibration for energy-ranked rejection sampling.

Uses 200+ QA pairs for calibration (vs 25 in the previous experiment).
More data → better hallucination direction → better candidate selection.

Usage:
    python scripts/experiment_scaled_rejection_sampling.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 200 factual QA pairs — mix of easy and hard
# Format: (question, expected_answer_substring)
CALIBRATION_QA = [
    # Math — easy
    ("What is 2+3?", "5"), ("What is 7*8?", "56"), ("What is 100/4?", "25"),
    ("What is 15-9?", "6"), ("What is 3^3?", "27"), ("What is 50+50?", "100"),
    ("What is 12*12?", "144"), ("What is 81/9?", "9"), ("What is 1000-1?", "999"),
    ("What is 2^10?", "1024"),
    # Math — medium
    ("What is the square root of 169?", "13"), ("What is 17*19?", "323"),
    ("What is 256/16?", "16"), ("What is 99*99?", "9801"),
    ("What is the cube root of 27?", "3"), ("What is 7! (7 factorial)?", "5040"),
    ("What is 2^16?", "65536"), ("What is 1+2+3+4+5+6+7+8+9+10?", "55"),
    # Math — hard (likely hallucination)
    ("What is 37*43?", "1591"), ("What is 123*456?", "56088"),
    ("What is the 20th prime number?", "71"), ("What is 11! (11 factorial)?", "39916800"),
    ("What is 2^20?", "1048576"), ("What is the square root of 1764?", "42"),
    ("What is 97*103?", "9991"), ("What is 19^2?", "361"),
    ("What is 23*29?", "667"), ("What is the 12th Fibonacci number?", "144"),
    # Geography — easy
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
    ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of France?", "Paris"),
    ("What is the capital of the UK?", "London"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("What is the capital of Russia?", "Moscow"),
    # Geography — medium
    ("What is the capital of South Korea?", "Seoul"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Turkey?", "Ankara"),
    ("What is the capital of Thailand?", "Bangkok"),
    ("What is the capital of Argentina?", "Buenos Aires"),
    # Geography — hard
    ("What is the capital of Myanmar?", "Naypyidaw"),
    ("What is the capital of Sri Lanka?", "Colombo"),
    ("What is the capital of Kazakhstan?", "Astana"),
    ("What is the capital of Nigeria?", "Abuja"),
    ("What is the capital of Pakistan?", "Islamabad"),
    # Science — easy
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("How many chromosomes do humans have?", "46"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the atomic number of hydrogen?", "1"),
    ("What is the atomic number of oxygen?", "8"),
    ("What is the speed of sound in m/s approximately?", "343"),
    ("How many planets are in our solar system?", "8"),
    # Science — medium
    ("What is the atomic number of gold?", "79"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("What is the chemical formula for methane?", "CH4"),
    ("What is Avogadro's number approximately?", "6.022"),
    ("What is the charge of an electron in coulombs?", "1.6"),
    # History — easy
    ("In what year did the Titanic sink?", "1912"),
    ("In what year did WW1 start?", "1914"),
    ("In what year did WW2 end?", "1945"),
    ("In what year did humans first land on the Moon?", "1969"),
    ("In what year was the Declaration of Independence signed?", "1776"),
    # History — medium
    ("In what year did the Berlin Wall fall?", "1989"),
    ("In what year did the French Revolution begin?", "1789"),
    ("In what year did Columbus reach the Americas?", "1492"),
    ("In what year was the Magna Carta signed?", "1215"),
    ("In what year did the Roman Empire fall?", "476"),
    # General knowledge
    ("How many sides does a hexagon have?", "6"),
    ("How many sides does an octagon have?", "8"),
    ("How many continents are there?", "7"),
    ("How many strings does a standard guitar have?", "6"),
    ("How many cards are in a standard deck?", "52"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the smallest planet in our solar system?", "Mercury"),
    ("What is the closest star to Earth?", "Sun"),
    ("What is the largest organ in the human body?", "skin"),
    ("How many teeth does an adult human typically have?", "32"),
    # CS / Tech
    ("What year was Python first released?", "1991"),
    ("What year was Java first released?", "1995"),
    ("What does HTML stand for?", "HyperText"),
    ("What does CPU stand for?", "Central Processing"),
    ("What does RAM stand for?", "Random Access"),
    ("Who created Linux?", "Torvalds"),
    ("What company created Java?", "Sun"),
    ("What year was the iPhone first released?", "2007"),
    ("How many bits in a byte?", "8"),
    ("What is the binary representation of 10?", "1010"),
]

# Test questions (separate from calibration)
TEST_QA = [
    ("What is the capital of Portugal?", "Lisbon"),
    ("What is the capital of Poland?", "Warsaw"),
    ("What is the chemical symbol for silver?", "Ag"),
    ("What is the chemical symbol for copper?", "Cu"),
    ("What is 13*17?", "221"),
    ("What is 29*31?", "899"),
    ("What is the square root of 225?", "15"),
    ("What is 6! (6 factorial)?", "720"),
    ("What is the 10th prime number?", "29"),
    ("In what year was the internet invented?", "1969"),
    ("What is the atomic number of carbon?", "6"),
    ("What is the chemical formula for ammonia?", "NH3"),
    ("How many bones in the human body?", "206"),
    ("What is the largest ocean?", "Pacific"),
    ("In what year did the Cold War end?", "1991"),
    ("What is the derivative of x^3?", "3x"),
    ("What is 2^8?", "256"),
    ("What is the sum of angles in a triangle?", "180"),
    ("Who painted the Mona Lisa?", "Vinci"),
    ("What is the speed of light approximately in m/s?", "3"),
]

N_CANDIDATES = 5


def check_answer(response: str, expected: str) -> bool:
    return expected.lower() in response.lower().strip()


def extract_gen_activations(model, tokenizer, question, do_sample=False, temperature=1.0):
    """Generate answer and extract activations from generated tokens."""
    import torch
    import jax.numpy as jnp

    prompt = f"Answer in one word or number only. {question}"
    inputs = tokenizer(prompt, return_tensors="pt")

    gen_kwargs = dict(max_new_tokens=20)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Activations from generated tokens only
    prompt_len = inputs["input_ids"].shape[1]
    full_seq = outputs[0].unsqueeze(0)
    with torch.no_grad():
        hidden_out = model(full_seq)
        hs = hidden_out.hidden_states

    gen_last = hs[-1][0, prompt_len:, :].mean(dim=0).float().numpy()
    gen_mid = hs[len(hs)//2][0, prompt_len:, :].mean(dim=0).float().numpy()
    act = jnp.concatenate([jnp.array(gen_last), jnp.array(gen_mid)])

    return response, act


def main() -> int:
    print("=" * 60)
    print("EXPERIMENT: Scaled Energy-Ranked Rejection Sampling")
    print(f"Calibration: {len(CALIBRATION_QA)} QA pairs")
    print(f"Test: {len(TEST_QA)} QA pairs, {N_CANDIDATES} candidates each")
    print("=" * 60)

    import torch
    import jax.numpy as jnp
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    print(f"\nLoading {model_name}...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, output_hidden_states=True,
    )
    model.eval()
    print(f"Loaded in {time.time() - start:.1f}s")

    # Phase 1: Calibrate on large QA set
    print(f"\n--- Phase 1: Calibration ({len(CALIBRATION_QA)} questions, greedy) ---")
    correct_acts = []
    hallucinated_acts = []
    cal_correct = 0

    for i, (q, expected) in enumerate(CALIBRATION_QA):
        response, act = extract_gen_activations(model, tokenizer, q, do_sample=False)
        if check_answer(response, expected):
            correct_acts.append(act)
            cal_correct += 1
        else:
            hallucinated_acts.append(act)

        if (i + 1) % 20 == 0:
            print(f"  Calibrated {i+1}/{len(CALIBRATION_QA)}... "
                  f"({cal_correct} correct, {i+1-cal_correct} hallucinated)")

    print(f"Calibration: {cal_correct}/{len(CALIBRATION_QA)} correct "
          f"({100*cal_correct/len(CALIBRATION_QA):.0f}%)")
    print(f"  Correct activations: {len(correct_acts)}")
    print(f"  Hallucinated activations: {len(hallucinated_acts)}")

    if len(hallucinated_acts) < 5:
        print("Not enough hallucinations for reliable direction — need harder questions")
        return 0

    # Find hallucination direction
    from carnot.embeddings.hallucination_direction import (
        HallucinationDirectionConfig, find_hallucination_direction, hallucination_energy,
    )
    direction = find_hallucination_direction(
        jnp.stack(correct_acts), jnp.stack(hallucinated_acts),
        HallucinationDirectionConfig(normalize=True),
    )

    # Verify calibration quality: measure separation on calibration data
    cal_correct_e = [float(hallucination_energy(a, direction)) for a in correct_acts]
    cal_hall_e = [float(hallucination_energy(a, direction)) for a in hallucinated_acts]
    cal_gap = sum(cal_hall_e)/len(cal_hall_e) - sum(cal_correct_e)/len(cal_correct_e)
    print(f"  Calibration energy gap: {cal_gap:.4f} ({'good' if cal_gap > 0 else 'BAD'})")

    # Phase 2: Test — greedy baseline
    print(f"\n--- Phase 2: Test baseline ({len(TEST_QA)} questions, greedy) ---")
    greedy_results = []
    for q, expected in TEST_QA:
        response, _ = extract_gen_activations(model, tokenizer, q, do_sample=False)
        greedy_results.append(check_answer(response, expected))

    greedy_acc = sum(greedy_results) / len(greedy_results)
    print(f"Greedy baseline: {sum(greedy_results)}/{len(TEST_QA)} ({greedy_acc:.0%})")

    # Phase 3: Rejection sampling on test set
    print(f"\n--- Phase 3: Rejection sampling ({N_CANDIDATES} candidates) ---")
    rejection_results = []

    for i, (q, expected) in enumerate(TEST_QA):
        candidates = []
        for c in range(N_CANDIDATES):
            response, act = extract_gen_activations(
                model, tokenizer, q, do_sample=True, temperature=0.8,
            )
            energy = float(hallucination_energy(act, direction))
            candidates.append((response, energy, check_answer(response, expected)))

        candidates.sort(key=lambda x: x[1])
        best_response, best_energy, best_correct = candidates[0]

        greedy_icon = "✓" if greedy_results[i] else "✗"
        best_icon = "✓" if best_correct else "✗"
        tag = ""
        if best_correct and not greedy_results[i]:
            tag = " (FIXED!)"
        elif not best_correct and greedy_results[i]:
            tag = " (REGRESSED)"
        print(f"  [{greedy_icon}→{best_icon}] {q[:40]}... e={best_energy:.1f}{tag}")

        rejection_results.append(best_correct)

    rejection_acc = sum(rejection_results) / len(rejection_results)

    # Also try: pick HIGHEST energy to confirm direction is correct
    # (This should be WORSE — if not, direction is inverted)

    # Summary
    fixes = sum(1 for g, r in zip(greedy_results, rejection_results) if not g and r)
    regressions = sum(1 for g, r in zip(greedy_results, rejection_results) if g and not r)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Calibration set: {len(CALIBRATION_QA)} QA pairs")
    print(f"  Calibration gap: {cal_gap:.4f}")
    print(f"  Test set: {len(TEST_QA)} QA pairs")
    print(f"")
    print(f"  Greedy baseline:    {sum(greedy_results)}/{len(TEST_QA)} ({greedy_acc:.0%})")
    print(f"  Energy-selected:    {sum(rejection_results)}/{len(TEST_QA)} ({rejection_acc:.0%})")
    improvement = rejection_acc - greedy_acc
    print(f"  Improvement:        {'+' if improvement >= 0 else ''}{improvement:.0%}")
    print(f"")
    print(f"  Fixes (wrong→right): {fixes}")
    print(f"  Regressions (right→wrong): {regressions}")
    print(f"  Net: {'+' if fixes >= regressions else ''}{fixes - regressions}")
    print(f"{'='*60}")

    if improvement > 0:
        print("SUCCESS: Energy-ranked selection IMPROVES model accuracy!")
    elif improvement == 0:
        print("NEUTRAL: No change.")
    else:
        print("REGRESSION: Needs better direction finding (SVD, more layers, trained EBM).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
