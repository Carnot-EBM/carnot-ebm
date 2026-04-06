#!/usr/bin/env python3
"""Experiment: Concept-specific steering on Qwen3-0.6B.

From Anthropic's emotion research: specific concept vectors (like "desperate")
are more effective than generic directions. Test whether steering with
"confabulation" specifically beats the generic hallucination direction.

Usage:
    python scripts/experiment_concept_steering.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Same questions as activation steering experiment
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
    ("What is the derivative of x squared?", "2x"),
    ("What gas do plants absorb?", "CO2"),
    ("What is the 15th prime number?", "47"),
    ("What is 17 * 23?", "391"),
    ("What is the integral of 1/x?", "ln"),
    ("How many bones in the adult human body?", "206"),
    ("What is the sum of angles in a pentagon?", "540"),
    ("What is the 8th Fibonacci number?", "21"),
]


def check_answer(response: str, expected: str) -> bool:
    return expected.lower() in response.lower().strip()


def steer_and_evaluate(model, tokenizer, questions, direction_torch, layers, alpha):
    """Run steered generation on all questions, return accuracy."""
    import torch

    results = []
    for q, expected in questions:
        prompt = f"Answer in one word or number only. {q}"
        inputs = tokenizer(prompt, return_tensors="pt")

        handles = []
        for layer_idx in layers:
            if hasattr(model.model, 'layers') and layer_idx < len(model.model.layers):
                def make_hook(d, a):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            hidden = output[0]
                            orig_dtype = hidden.dtype
                            dv = d[:hidden.shape[-1]].to(device=hidden.device).float()
                            modified = (hidden.float() + a * dv).to(orig_dtype)
                            return (modified,) + output[1:]
                        return output
                    return hook_fn
                h = model.model.layers[layer_idx].register_forward_hook(
                    make_hook(direction_torch, alpha)
                )
                handles.append(h)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        for h in handles:
            h.remove()

        results.append(check_answer(response, expected))

    return sum(results) / len(results), results


def main() -> int:
    import torch
    import jax.numpy as jnp
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, output_hidden_states=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"Layers: {n_layers}")

    # Phase 1: Greedy baseline
    print("\n--- Phase 1: Greedy baseline ---")
    greedy_results = []
    correct_acts, wrong_acts = [], []

    for q, expected in QUESTIONS:
        prompt = f"Answer in one word or number only. {q}"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            hidden = model(outputs[0].unsqueeze(0))
            hs = hidden.hidden_states
        act = hs[-1][0, prompt_len:, :].mean(dim=0).float().numpy()

        correct = check_answer(response, expected)
        greedy_results.append(correct)
        if correct:
            correct_acts.append(jnp.array(act))
        else:
            wrong_acts.append(jnp.array(act))

    greedy_acc = sum(greedy_results) / len(greedy_results)
    print(f"Greedy: {sum(greedy_results)}/{len(QUESTIONS)} ({greedy_acc:.0%})")

    if len(wrong_acts) < 3:
        print("Not enough wrong answers")
        return 0

    # Phase 2: Compute concept vectors
    print("\n--- Phase 2: Concept vectors ---")
    from carnot.embeddings.hallucination_direction import (
        HallucinationDirectionConfig, find_hallucination_direction,
    )

    # Generic direction
    generic_dir = find_hallucination_direction(
        jnp.stack(correct_acts), jnp.stack(wrong_acts),
        HallucinationDirectionConfig(normalize=True),
    )
    generic_torch = torch.tensor(generic_dir.tolist(), dtype=torch.float32)

    # For concept-specific vectors, we'd ideally use concept_vectors.py with
    # real model generation. As a simpler approach: use the WRONG activations
    # as "confabulation" and try different subsets of the direction.
    # We'll test: suppress at different layers and different alphas.

    # Phase 3: Test steering with different configs
    print("\n--- Phase 3: Steering experiments ---")
    best_layers = list(range(n_layers // 2, n_layers))[:5]  # upper-half layers
    mid_layers = list(range(n_layers // 4, 3 * n_layers // 4))[:5]
    all_layers = list(range(n_layers))

    configs = [
        ("Upper layers, alpha=-1.0", best_layers, -1.0),
        ("Upper layers, alpha=-0.5", best_layers, -0.5),
        ("Upper layers, alpha=-2.0", best_layers, -2.0),
        ("Mid layers, alpha=-1.0", mid_layers, -1.0),
        ("All layers, alpha=-0.5", all_layers, -0.5),
        ("Upper layers, alpha=+1.0 (amplify)", best_layers, 1.0),
    ]

    results_table = []
    for name, layers, alpha in configs:
        acc, _ = steer_and_evaluate(model, tokenizer, QUESTIONS, generic_torch, layers, alpha)
        delta = acc - greedy_acc
        results_table.append((name, acc, delta))
        print(f"  {name}: {acc:.0%} ({'+' if delta >= 0 else ''}{delta:.0%})")

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  {'Config':<40} {'Acc':>5} {'Δ':>6}")
    print(f"  {'─'*52}")
    print(f"  {'Greedy baseline':<40} {greedy_acc:>4.0%}   {'—':>5}")
    for name, acc, delta in results_table:
        print(f"  {name:<40} {acc:>4.0%}  {'+' if delta >= 0 else ''}{delta:>4.0%}")

    best = max(results_table, key=lambda r: r[1])
    print(f"\n  Best: {best[0]} at {best[1]:.0%} ({'+' if best[2] >= 0 else ''}{best[2]:.0%})")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
