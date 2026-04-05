#!/usr/bin/env python3
"""Experiment: Logprob-based rejection sampling.

No calibration, no training, no PCA — just use the model's own
per-token log-probabilities as energy. Generate N candidates, pick the
one with highest total logprob (lowest energy = most confident).

This is the semantic energy from arxiv 2508.14496, applied directly.

Usage:
    python scripts/experiment_logprob_rejection.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from experiment_scaled_rejection_sampling import CALIBRATION_QA, TEST_QA

N_CANDIDATES = 5


def check_answer(response: str, expected: str) -> bool:
    return expected.lower() in response.lower().strip()


def generate_with_logprobs(model, tokenizer, question, do_sample=False, temperature=1.0):
    """Generate answer and compute total logprob of generated tokens."""
    import torch

    prompt = f"Answer in one word or number only. {question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(max_new_tokens=20)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, **gen_kwargs,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0, prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute logprobs from scores
    total_logprob = 0.0
    n_tokens = 0
    if hasattr(outputs, "scores") and outputs.scores:
        for step_idx, scores in enumerate(outputs.scores):
            if step_idx >= len(generated_ids):
                break
            token_id = generated_ids[step_idx].item()
            log_probs = torch.log_softmax(scores[0], dim=-1)
            total_logprob += log_probs[token_id].item()
            n_tokens += 1

    # Mean logprob per token (normalize for length)
    mean_logprob = total_logprob / max(n_tokens, 1)

    return response, total_logprob, mean_logprob, n_tokens


def main() -> int:
    print("=" * 60)
    print("EXPERIMENT: Logprob-Based Rejection Sampling")
    print("No calibration, no training — pure model confidence")
    print(f"Candidates per question: {N_CANDIDATES}")
    print("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-0.6B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    print("Loaded.")

    # Phase 1: Greedy baseline + logprob analysis
    print(f"\n--- Phase 1: Greedy baseline + logprob sanity check ---")
    greedy_results = []
    correct_logprobs = []
    wrong_logprobs = []

    for q, expected in TEST_QA:
        response, total_lp, mean_lp, n_tok = generate_with_logprobs(model, tokenizer, q)
        correct = check_answer(response, expected)
        greedy_results.append(correct)
        if correct:
            correct_logprobs.append(mean_lp)
        else:
            wrong_logprobs.append(mean_lp)

    greedy_acc = sum(greedy_results) / len(greedy_results)
    print(f"Greedy: {sum(greedy_results)}/{len(TEST_QA)} ({greedy_acc:.0%})")

    if correct_logprobs and wrong_logprobs:
        mean_c = sum(correct_logprobs) / len(correct_logprobs)
        mean_w = sum(wrong_logprobs) / len(wrong_logprobs)
        print(f"Mean logprob (correct): {mean_c:.3f}")
        print(f"Mean logprob (wrong):   {mean_w:.3f}")
        print(f"Gap: {mean_c - mean_w:.3f} ({'correct > wrong = good signal' if mean_c > mean_w else 'wrong > correct = BAD'})")

    # Phase 2: Rejection sampling — pick highest mean logprob
    print(f"\n--- Phase 2: Rejection sampling (best of {N_CANDIDATES}) ---")
    rejection_results = []

    for i, (q, expected) in enumerate(TEST_QA):
        candidates = []
        for c in range(N_CANDIDATES):
            response, total_lp, mean_lp, n_tok = generate_with_logprobs(
                model, tokenizer, q, do_sample=True, temperature=0.8,
            )
            candidates.append((response, mean_lp, check_answer(response, expected)))

        # Pick HIGHEST logprob (most confident = lowest energy)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_response, best_lp, best_correct = candidates[0]

        # Also: did any candidate get it right?
        any_correct = any(c[2] for c in candidates)

        greedy_icon = "✓" if greedy_results[i] else "✗"
        best_icon = "✓" if best_correct else "✗"
        tag = ""
        if best_correct and not greedy_results[i]:
            tag = " ★ FIXED"
        elif not best_correct and greedy_results[i]:
            tag = " ✖ REG"
        oracle = f" (oracle: {sum(c[2] for c in candidates)}/{N_CANDIDATES})" if any_correct else ""

        print(f"  [{greedy_icon}→{best_icon}] {q[:40]}... lp={best_lp:.2f}{tag}{oracle}")

        rejection_results.append(best_correct)

    rejection_acc = sum(rejection_results) / len(rejection_results)
    fixes = sum(1 for g, r in zip(greedy_results, rejection_results) if not g and r)
    regressions = sum(1 for g, r in zip(greedy_results, rejection_results) if g and not r)

    # Also compute oracle (best possible if we always picked a correct candidate)
    # This is the upper bound — how often at least one of N candidates is correct
    oracle_results = []
    for i, (q, expected) in enumerate(TEST_QA):
        # Reuse — we need to regenerate for oracle
        pass

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Greedy baseline:    {sum(greedy_results)}/{len(TEST_QA)} ({greedy_acc:.0%})")
    print(f"  Logprob-selected:   {sum(rejection_results)}/{len(TEST_QA)} ({rejection_acc:.0%})")
    improvement = rejection_acc - greedy_acc
    print(f"  Improvement:        {'+' if improvement >= 0 else ''}{improvement:.0%}")
    print(f"")
    print(f"  Fixes: {fixes}, Regressions: {regressions}, Net: {'+' if fixes>=regressions else ''}{fixes-regressions}")
    print(f"{'='*60}")

    if improvement > 0:
        print("SUCCESS: Logprob selection improves accuracy!")
        print("The model's own confidence IS the right energy signal.")
    elif improvement == 0:
        print("NEUTRAL: Same accuracy.")
    else:
        print("REGRESSION: Logprob confidence doesn't correlate with correctness here.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
