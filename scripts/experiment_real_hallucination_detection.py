#!/usr/bin/env python3
"""Experiment: Real hallucination detection via activation-space energy.

Downloads a small local model (Qwen3-0.6B), asks it factual questions,
extracts per-layer activations, finds the hallucination direction, and
measures whether the energy function separates correct from hallucinated
answers.

This is the first test where synthetic mocks meet reality.

Usage:
    python scripts/experiment_real_hallucination_detection.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Factual questions with known answers — mix of easy (model gets right)
# and tricky (model may hallucinate)
QUESTIONS = [
    # Easy facts (likely correct)
    ("What is the capital of France?", "Paris", "factual"),
    ("What is 2 + 2?", "4", "math"),
    ("What color is the sky on a clear day?", "blue", "factual"),
    ("How many legs does a dog have?", "4", "factual"),
    ("What is the chemical symbol for water?", "H2O", "factual"),
    ("What planet is closest to the Sun?", "Mercury", "factual"),
    ("What is the square root of 144?", "12", "math"),
    ("In what year did World War II end?", "1945", "factual"),
    ("What is the largest ocean on Earth?", "Pacific", "factual"),
    ("How many days are in a week?", "7", "factual"),
    # Medium difficulty
    ("What is the atomic number of carbon?", "6", "science"),
    ("Who wrote Romeo and Juliet?", "Shakespeare", "factual"),
    ("What is the speed of light in km/s approximately?", "300000", "science"),
    ("What is the derivative of x squared?", "2x", "math"),
    ("What gas do plants absorb from the atmosphere?", "CO2", "science"),
    # Harder / more likely to hallucinate
    ("What is the 15th prime number?", "47", "math"),
    ("What is 17 * 23?", "391", "math"),
    ("What is the integral of 1/x?", "ln|x|", "math"),
    ("What is the population of Iceland approximately?", "380000", "factual"),
    ("How many bones are in the adult human body?", "206", "science"),
    ("What is the sum of angles in a pentagon?", "540", "math"),
    ("What is 13 factorial?", "6227020800", "math"),
    ("What is the 8th Fibonacci number?", "21", "math"),
    ("What is the chemical formula for glucose?", "C6H12O6", "science"),
    ("What year was the Python programming language first released?", "1991", "factual"),
]


def check_answer(response: str, expected: str) -> bool:
    """Check if the model's response contains the expected answer."""
    response_lower = response.lower().strip()
    expected_lower = expected.lower().strip()
    return expected_lower in response_lower


def main() -> int:
    print("=" * 60)
    print("EXPERIMENT: Real Hallucination Detection via Activation Energy")
    print("=" * 60)

    # Step 1: Load model
    print("\n--- Step 1: Loading model ---")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers torch")
        return 1

    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - start:.1f}s")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Step 2: Ask questions and extract activations
    print("\n--- Step 2: Asking questions and extracting activations ---")
    import torch
    import jax.numpy as jnp

    correct_activations = []
    hallucinated_activations = []
    results = []

    for i, (question, expected, category) in enumerate(QUESTIONS):
        prompt = f"Answer in one word or number only. {question}"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Get hidden states for the full input
        with torch.no_grad():
            hidden_outputs = model(**inputs)
            hidden_states = hidden_outputs.hidden_states  # tuple of (batch, seq, dim) per layer

        # Extract mean activation from last layer as the representation
        last_hidden = hidden_states[-1][0].mean(dim=0).float().numpy()  # (dim,)
        mid_hidden = hidden_states[len(hidden_states) // 2][0].mean(dim=0).float().numpy()

        # Concatenate last + mid layer as feature vector
        activation = jnp.concatenate([
            jnp.array(last_hidden),
            jnp.array(mid_hidden),
        ])

        is_correct = check_answer(response, expected)
        status = "✓" if is_correct else "✗"

        if is_correct:
            correct_activations.append(activation)
        else:
            hallucinated_activations.append(activation)

        results.append({
            "question": question,
            "expected": expected,
            "response": response.strip()[:50],
            "correct": is_correct,
            "category": category,
        })

        print(f"  [{status}] Q: {question[:40]}... → {response.strip()[:30]}")

    n_correct = sum(1 for r in results if r["correct"])
    n_wrong = len(results) - n_correct
    print(f"\nModel accuracy: {n_correct}/{len(results)} ({100*n_correct/len(results):.0f}%)")
    print(f"Correct: {n_correct}, Hallucinated: {n_wrong}")

    if n_wrong == 0:
        print("\nModel got everything right — no hallucinations to detect!")
        print("Try harder questions or a weaker model.")
        return 0

    if n_correct == 0:
        print("\nModel got everything wrong — can't find correct pattern!")
        return 1

    # Step 3: Find hallucination direction
    print("\n--- Step 3: Finding hallucination direction ---")
    from carnot.embeddings.hallucination_direction import (
        HallucinationDirectionConfig,
        find_hallucination_direction,
        hallucination_energy,
    )

    correct_batch = jnp.stack(correct_activations)
    hallucinated_batch = jnp.stack(hallucinated_activations)

    print(f"Correct activations: {correct_batch.shape}")
    print(f"Hallucinated activations: {hallucinated_batch.shape}")

    config = HallucinationDirectionConfig(normalize=True)
    direction = find_hallucination_direction(correct_batch, hallucinated_batch, config)
    print(f"Hallucination direction: shape={direction.shape}, norm={float(jnp.linalg.norm(direction)):.4f}")

    # Step 4: Compute energy and measure separation
    print("\n--- Step 4: Measuring energy separation ---")
    correct_energies = [float(hallucination_energy(a, direction)) for a in correct_activations]
    hallucinated_energies = [float(hallucination_energy(a, direction)) for a in hallucinated_activations]

    mean_correct = sum(correct_energies) / len(correct_energies)
    mean_hallucinated = sum(hallucinated_energies) / len(hallucinated_energies)
    gap = mean_hallucinated - mean_correct

    print(f"Mean energy (correct):      {mean_correct:.4f}")
    print(f"Mean energy (hallucinated): {mean_hallucinated:.4f}")
    print(f"Energy gap:                 {gap:.4f}")

    # Classification accuracy: threshold at midpoint
    threshold = (mean_correct + mean_hallucinated) / 2
    tp = sum(1 for e in hallucinated_energies if e > threshold)
    tn = sum(1 for e in correct_energies if e <= threshold)
    accuracy = (tp + tn) / (len(correct_energies) + len(hallucinated_energies))

    print(f"\nClassification (threshold={threshold:.4f}):")
    print(f"  True positives (hallucination detected):  {tp}/{len(hallucinated_energies)}")
    print(f"  True negatives (correct confirmed):       {tn}/{len(correct_energies)}")
    print(f"  Accuracy: {accuracy:.1%}")

    # Step 5: Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Questions: {len(QUESTIONS)}")
    print(f"  Model accuracy: {n_correct}/{len(results)} ({100*n_correct/len(results):.0f}%)")
    print(f"  Energy gap: {gap:.4f} ({'POSITIVE — hallucinations have higher energy' if gap > 0 else 'NEGATIVE — direction is inverted'})")
    print(f"  Detection accuracy: {accuracy:.1%}")
    print(f"{'='*60}")

    if gap > 0 and accuracy > 0.6:
        print("SUCCESS: The energy function separates correct from hallucinated!")
        print("The activation-space EBM detects hallucinations the LLM cannot self-detect.")
    elif gap > 0:
        print("PARTIAL: Energy gap is positive but detection accuracy is low.")
        print("Need more data or better layer selection.")
    else:
        print("NEGATIVE: Energy gap is inverted. The direction may need refinement.")
        print("Try using more layers or SVD with top-k directions.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
