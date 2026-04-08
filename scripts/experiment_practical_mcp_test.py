#!/usr/bin/env python3
"""Practical deployment test: use the MCP tools to detect a real hallucination.

Simulates what happens when Claude Code uses the Carnot MCP server:
1. Generate a response from Qwen3.5-0.8B on a tricky question
2. Extract activations
3. Score with the pre-trained EBM via the same API the MCP server uses
4. Compare the EBM's assessment with ground truth

This tests the full end-to-end pipeline: model → activations → EBM → verdict.

Usage:
    .venv/bin/python scripts/experiment_practical_mcp_test.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from carnot.inference.ebm_loader import load_ebm
    from carnot.inference.ebm_rejection import score_activations_with_ebm
    import jax.numpy as jnp

    print("=" * 70)
    print("PRACTICAL DEPLOYMENT TEST: End-to-End Hallucination Detection")
    print("=" * 70)

    # Load model and EBM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3.5-0.8B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        output_hidden_states=True,
        torch_dtype=torch.float16 if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    print("Loading pre-trained EBM...")
    ebm = load_ebm("per-token-ebm-qwen35-08b-nothink")

    # Test questions — mix of easy (should be correct) and tricky (likely to hallucinate)
    test_cases = [
        {
            "question": "What is the capital of France?",
            "expected": "Paris",
            "difficulty": "easy",
        },
        {
            "question": "What is 2 + 2?",
            "expected": "4",
            "difficulty": "easy",
        },
        {
            "question": "Who was the first person to walk on Mars?",
            "expected": None,  # No one has — should say "no one"
            "difficulty": "trick",
        },
        {
            "question": "What year did the Berlin Wall fall?",
            "expected": "1989",
            "difficulty": "medium",
        },
        {
            "question": "What is the population of the fictional city of Atlantis?",
            "expected": None,  # Fictional — should say unknown/fictional
            "difficulty": "trick",
        },
        {
            "question": "Who wrote the novel 'The Great Gatsby'?",
            "expected": "Fitzgerald",
            "difficulty": "easy",
        },
        {
            "question": "What is the speed of dark?",
            "expected": None,  # Nonsensical — should say it's not a real concept
            "difficulty": "trick",
        },
        {
            "question": "In what year did Napoleon conquer Australia?",
            "expected": None,  # Never happened
            "difficulty": "trick",
        },
    ]

    results = []
    print(f"\nTesting {len(test_cases)} questions...\n")

    for i, tc in enumerate(test_cases):
        question = tc["question"]
        expected = tc["expected"]
        difficulty = tc["difficulty"]

        # Generate response
        prompt = f"Answer briefly and factually. {question}"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)

        gen_ids = outputs[0, prompt_len:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        # Extract activations and score with EBM
        with torch.no_grad():
            ho = model(outputs, output_hidden_states=True)
            hs = ho.hidden_states

        last_layer = hs[-1][0, prompt_len:, :].float().cpu().numpy()
        mean_energy = score_activations_with_ebm(ebm, last_layer)

        # Check correctness
        if expected is not None:
            is_correct = expected.lower() in response.lower()
        else:
            # For trick questions, check if model hedges/refuses
            hedge_words = ["no one", "never", "fictional", "not real", "doesn't exist",
                           "hasn't", "no person", "nobody", "not a real", "impossible",
                           "no such", "didn't", "not possible"]
            is_correct = any(w in response.lower() for w in hedge_words)

        # EBM verdict: lower energy = more likely correct
        # Use a simple threshold (this is the practical question: what threshold?)
        ebm_says_ok = mean_energy < 0  # Rough heuristic

        results.append({
            "question": question,
            "response": response[:100],
            "difficulty": difficulty,
            "actually_correct": is_correct,
            "mean_energy": mean_energy,
            "ebm_says_ok": ebm_says_ok,
            "ebm_agrees": (ebm_says_ok == is_correct),
            "n_tokens": len(gen_ids),
        })

        icon = "CORRECT" if is_correct else "WRONG"
        ebm_icon = "low" if ebm_says_ok else "HIGH"
        agree = "agree" if (ebm_says_ok == is_correct) else "DISAGREE"
        print(f"  Q{i+1} [{difficulty:6s}] {icon:7s} energy={mean_energy:+.3f} ({ebm_icon}) [{agree}]")
        print(f"    Q: {question}")
        print(f"    A: {response[:80]}")
        print()

    # Summary
    sep = "=" * 70
    print(sep)
    print("PRACTICAL TEST RESULTS")
    print(sep)

    n_correct = sum(1 for r in results if r["actually_correct"])
    n_ebm_agree = sum(1 for r in results if r["ebm_agrees"])
    n_total = len(results)

    correct_energies = [r["mean_energy"] for r in results if r["actually_correct"]]
    wrong_energies = [r["mean_energy"] for r in results if not r["actually_correct"]]

    print(f"  Model accuracy: {n_correct}/{n_total}")
    print(f"  EBM agreement:  {n_ebm_agree}/{n_total} ({n_ebm_agree/n_total:.0%})")

    if correct_energies:
        print(f"  Correct answer mean energy: {np.mean(correct_energies):+.4f}")
    if wrong_energies:
        print(f"  Wrong answer mean energy:   {np.mean(wrong_energies):+.4f}")
    if correct_energies and wrong_energies:
        gap = np.mean(wrong_energies) - np.mean(correct_energies)
        print(f"  Energy gap (wrong - correct): {gap:+.4f}")

    print(f"\n  By difficulty:")
    for diff in ["easy", "medium", "trick"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            model_acc = sum(1 for r in subset if r["actually_correct"]) / len(subset)
            ebm_agree = sum(1 for r in subset if r["ebm_agrees"]) / len(subset)
            print(f"    {diff:8s}: model={model_acc:.0%}, ebm_agree={ebm_agree:.0%}")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
