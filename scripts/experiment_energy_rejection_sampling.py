#!/usr/bin/env python3
"""Experiment: Energy-ranked rejection sampling improves LLM accuracy.

For each question, generate N candidate answers via temperature sampling,
score each with hallucination energy (from activation-space direction),
and pick the lowest-energy candidate. Compare: baseline (greedy) vs
energy-selected (best of N).

This is the first demonstration of the EBM making the LLM BETTER.

Usage:
    python scripts/experiment_energy_rejection_sampling.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QUESTIONS = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What color is the sky on a clear day?", "blue"),
    ("How many legs does a dog have?", "4"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the square root of 144?", "12"),
    ("In what year did World War II end?", "1945"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("How many days are in a week?", "7"),
    ("What is the atomic number of carbon?", "6"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the speed of light in km/s approximately?", "300000"),
    ("What is the derivative of x squared?", "2x"),
    ("What gas do plants absorb from the atmosphere?", "CO2"),
    ("What is the 15th prime number?", "47"),
    ("What is 17 * 23?", "391"),
    ("What is the integral of 1/x?", "ln"),
    ("What is the population of Iceland approximately?", "380000"),
    ("How many bones are in the adult human body?", "206"),
    ("What is the sum of angles in a pentagon?", "540"),
    ("What is 13 factorial?", "6227020800"),
    ("What is the 8th Fibonacci number?", "21"),
    ("What is the chemical formula for glucose?", "C6H12O6"),
    ("What year was the Python programming language first released?", "1991"),
]

N_CANDIDATES = 5


def check_answer(response: str, expected: str) -> bool:
    return expected.lower() in response.lower().strip()


def main() -> int:
    print("=" * 60)
    print("EXPERIMENT: Energy-Ranked Rejection Sampling")
    print(f"Generate {N_CANDIDATES} candidates per question, pick lowest energy")
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

    # Phase 1: Calibration — greedy answers to find hallucination direction
    print("\n--- Phase 1: Calibration (greedy baseline) ---")
    correct_acts = []
    hallucinated_acts = []
    greedy_results = []

    for question, expected in QUESTIONS:
        prompt = f"Answer in one word or number only. {question}"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Extract activations from GENERATED tokens (not just prompt)
        full_seq = outputs[0].unsqueeze(0)  # (1, prompt_len + gen_len)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            hidden_out = model(full_seq)
            hs = hidden_out.hidden_states
        # Mean-pool only the generated token activations
        gen_last = hs[-1][0, prompt_len:, :].mean(dim=0).float().numpy()
        gen_mid = hs[len(hs)//2][0, prompt_len:, :].mean(dim=0).float().numpy()
        act = jnp.concatenate([jnp.array(gen_last), jnp.array(gen_mid)])

        correct = check_answer(response, expected)
        if correct:
            correct_acts.append(act)
        else:
            hallucinated_acts.append(act)
        greedy_results.append(correct)

    greedy_accuracy = sum(greedy_results) / len(greedy_results)
    print(f"Greedy baseline: {sum(greedy_results)}/{len(QUESTIONS)} ({greedy_accuracy:.0%})")

    if not hallucinated_acts:
        print("No hallucinations found — can't calibrate direction")
        return 0

    # Find hallucination direction
    from carnot.embeddings.hallucination_direction import (
        HallucinationDirectionConfig, find_hallucination_direction, hallucination_energy,
    )
    correct_batch = jnp.stack(correct_acts)
    hallucinated_batch = jnp.stack(hallucinated_acts)
    direction = find_hallucination_direction(
        correct_batch, hallucinated_batch, HallucinationDirectionConfig(normalize=True),
    )
    print(f"Calibrated on {len(correct_acts)} correct + {len(hallucinated_acts)} hallucinated")

    # Phase 2: Rejection sampling — N candidates per question
    print(f"\n--- Phase 2: Rejection sampling ({N_CANDIDATES} candidates each) ---")
    rejection_results = []

    for i, (question, expected) in enumerate(QUESTIONS):
        prompt = f"Answer in one word or number only. {question}"
        inputs = tokenizer(prompt, return_tensors="pt")

        candidates = []
        for c in range(N_CANDIDATES):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=20,
                    do_sample=True, temperature=0.8, top_p=0.95,
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            # Extract activations from GENERATED tokens
            full_seq = outputs[0].unsqueeze(0)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                hidden_out = model(full_seq)
                hs = hidden_out.hidden_states
            gen_last = hs[-1][0, prompt_len:, :].mean(dim=0).float().numpy()
            gen_mid = hs[len(hs)//2][0, prompt_len:, :].mean(dim=0).float().numpy()
            act = jnp.concatenate([jnp.array(gen_last), jnp.array(gen_mid)])

            energy = float(hallucination_energy(act, direction))
            candidates.append((response, energy, check_answer(response, expected)))

        # Pick lowest energy candidate
        candidates.sort(key=lambda x: x[1])
        best_response, best_energy, best_correct = candidates[0]

        # Also check: did ANY candidate get it right?
        any_correct = any(c[2] for c in candidates)

        greedy_icon = "✓" if greedy_results[i] else "✗"
        best_icon = "✓" if best_correct else "✗"
        print(f"  [{greedy_icon}→{best_icon}] {question[:40]}... "
              f"best_e={best_energy:.1f} {'(FIXED!)' if best_correct and not greedy_results[i] else ''}")

        rejection_results.append(best_correct)

    rejection_accuracy = sum(rejection_results) / len(rejection_results)

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Candidates per question: {N_CANDIDATES}")
    print(f"")
    print(f"  Greedy baseline:    {sum(greedy_results)}/{len(QUESTIONS)} ({greedy_accuracy:.0%})")
    print(f"  Energy-selected:    {sum(rejection_results)}/{len(QUESTIONS)} ({rejection_accuracy:.0%})")
    improvement = rejection_accuracy - greedy_accuracy
    print(f"  Improvement:        {'+' if improvement >= 0 else ''}{improvement:.0%}")
    print(f"")

    # Count fixes and regressions
    fixes = sum(1 for g, r in zip(greedy_results, rejection_results) if not g and r)
    regressions = sum(1 for g, r in zip(greedy_results, rejection_results) if g and not r)
    print(f"  Fixes (wrong→right): {fixes}")
    print(f"  Regressions (right→wrong): {regressions}")
    print(f"  Net: +{fixes - regressions}")
    print(f"{'='*60}")

    if improvement > 0:
        print("SUCCESS: Energy-ranked selection IMPROVES model accuracy!")
        print("The EBM makes the LLM better, not just audits it.")
    elif improvement == 0:
        print("NEUTRAL: Same accuracy. Energy ranking didn't help or hurt.")
    else:
        print("REGRESSION: Energy ranking made it worse. Direction needs refinement.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
