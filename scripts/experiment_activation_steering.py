#!/usr/bin/env python3
"""Experiment: In-generation activation steering on Qwen3-0.6B.

THE critical experiment: does steering DURING generation beat post-hoc
scoring? Previous experiments showed post-hoc activation scoring doesn't
beat logprobs. Steering operates at the token level where the signal lives.

Usage:
    python scripts/experiment_activation_steering.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
    ("What is the chemical formula for glucose?", "C6H12O6"),
    ("What year was Python first released?", "1991"),
    ("What is 2^10?", "1024"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("How many continents are there?", "7"),
]


def check_answer(response: str, expected: str) -> bool:
    return expected.lower() in response.lower().strip()


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

    # Phase 1: Greedy baseline + collect activations for direction
    print("\n--- Phase 1: Greedy baseline ---")
    correct_acts, wrong_acts = [], []
    greedy_results = []

    for q, expected in QUESTIONS:
        prompt = f"Answer in one word or number only. {q}"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Get activations from generated tokens
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
    print(f"Correct: {len(correct_acts)}, Wrong: {len(wrong_acts)}")

    if len(wrong_acts) < 3:
        print("Not enough wrong answers to compute direction")
        return 0

    # Phase 2: Find hallucination direction
    print("\n--- Phase 2: Hallucination direction ---")
    from carnot.embeddings.hallucination_direction import (
        HallucinationDirectionConfig, find_hallucination_direction,
    )
    direction_jax = find_hallucination_direction(
        jnp.stack(correct_acts), jnp.stack(wrong_acts),
        HallucinationDirectionConfig(normalize=True),
    )
    direction_torch = torch.tensor(direction_jax.tolist(), dtype=torch.float32)
    print(f"Direction shape: {direction_torch.shape}")

    # Phase 3: Find best layers for steering
    print("\n--- Phase 3: Finding steerable layers ---")
    # Get number of layers
    if hasattr(model.model, 'layers'):
        n_layers = len(model.model.layers)
    elif hasattr(model.transformer, 'h'):
        n_layers = len(model.transformer.h)
    else:
        n_layers = model.config.num_hidden_layers
    print(f"Model has {n_layers} layers")

    # Score a few layers by perturbation impact
    test_layers = list(range(0, n_layers, max(1, n_layers // 8)))  # Sample ~8 layers
    layer_scores = {}

    for layer_idx in test_layers:
        prompt = f"Answer in one word or number only. {QUESTIONS[0][0]}"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Get baseline logits
        with torch.no_grad():
            base_out = model(**inputs)
            base_logits = base_out.logits[0, -1, :]

        # Get perturbed logits
        hook_handle = None
        def make_hook(direction, alpha=1.0):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                    # Broadcast direction to match hidden state shape
                    d = direction[:hidden.shape[-1]].to(hidden.device)
                    return (hidden + alpha * d,) + output[1:]
                return output + alpha * direction[:output.shape[-1]].to(output.device)
            return hook_fn

        if hasattr(model.model, 'layers'):
            hook_handle = model.model.layers[layer_idx].register_forward_hook(
                make_hook(direction_torch, alpha=2.0)
            )
        with torch.no_grad():
            pert_out = model(**inputs)
            pert_logits = pert_out.logits[0, -1, :]

        if hook_handle:
            hook_handle.remove()

        score = float(torch.norm(pert_logits - base_logits).item())
        layer_scores[layer_idx] = score

    best_layers = sorted(layer_scores, key=lambda l: layer_scores[l], reverse=True)[:3]
    print(f"Layer scores: {layer_scores}")
    print(f"Best layers: {best_layers}")

    # Phase 4: Steered generation
    print(f"\n--- Phase 4: Steered generation (layers {best_layers}) ---")
    steered_results = []

    for qi, (q, expected) in enumerate(QUESTIONS):
        prompt = f"Answer in one word or number only. {q}"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Register hooks on best layers
        handles = []
        for layer_idx in best_layers:
            if hasattr(model.model, 'layers'):
                h = model.model.layers[layer_idx].register_forward_hook(
                    make_hook(direction_torch, alpha=-1.0)  # SUBTRACT direction
                )
                handles.append(h)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        for h in handles:
            h.remove()

        correct = check_answer(response, expected)
        steered_results.append(correct)

        g_icon = "✓" if greedy_results[qi] else "✗"
        s_icon = "✓" if correct else "✗"
        tag = ""
        if correct and not greedy_results[qi]: tag = " ★ FIXED"
        elif not correct and greedy_results[qi]: tag = " ✖ REG"
        print(f"  [{g_icon}→{s_icon}] {q[:40]}...{tag}")

    steered_acc = sum(steered_results) / len(steered_results)
    fixes = sum(1 for g, s in zip(greedy_results, steered_results) if not g and s)
    regressions = sum(1 for g, s in zip(greedy_results, steered_results) if g and not s)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Greedy:   {sum(greedy_results)}/{len(QUESTIONS)} ({greedy_acc:.0%})")
    print(f"  Steered:  {sum(steered_results)}/{len(QUESTIONS)} ({steered_acc:.0%})")
    print(f"  Δ:        {'+' if steered_acc >= greedy_acc else ''}{steered_acc - greedy_acc:.0%}")
    print(f"  Fixes: {fixes}, Regressions: {regressions}, Net: {fixes - regressions}")
    print(f"{'='*60}")

    if steered_acc > greedy_acc:
        print("SUCCESS: In-generation steering improves accuracy!")
    elif steered_acc == greedy_acc:
        print("NEUTRAL: No change from steering.")
    else:
        print("REGRESSION: Steering hurt. Try different alpha or layers.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
