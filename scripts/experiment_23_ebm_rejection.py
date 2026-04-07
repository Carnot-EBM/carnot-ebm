#!/usr/bin/env python3
"""Experiment 23: EBM-guided rejection sampling on TruthfulQA.

Combines per-token EBM energy + logprob scores for candidate selection.
Tests whether the two signals are complementary.

Comparison:
  - Greedy (baseline): single deterministic generation
  - Logprob-only: N=5 candidates, select highest logprob
  - EBM-only: N=5 candidates, select lowest mean EBM energy
  - Composite: N=5 candidates, select lowest composite (ebm + logprob)

Usage:
    sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        .venv/bin/python scripts/experiment_23_ebm_rejection.py'
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def main() -> int:
    import torch
    import jax.numpy as jnp
    import jax.random as jrandom
    from datasets import load_dataset
    from safetensors.numpy import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss
    from carnot.inference.ebm_rejection import (
        EBMRejectionConfig,
        score_activations_with_ebm,
    )
    import jax

    # --- Step 1: Train EBM on Qwen3.5-0.8B activations ---
    print("=" * 60)
    print("EXPERIMENT 23: EBM-Guided Rejection Sampling")
    print("=" * 60)

    data_file = os.path.join(os.path.dirname(__file__), "..", "data", "token_activations_qwen35_merged.safetensors")
    if not os.path.exists(data_file):
        print(f"ERROR: {data_file} not found. Run merge_activations_qwen35.py first.")
        return 1

    print("\nStep 1: Training per-token EBM...")
    data = load_file(data_file)
    activations = jnp.array(data["activations"])
    labels = data["labels"]

    correct_mask = labels == 1
    wrong_mask = labels == 0
    correct_acts = activations[correct_mask]
    wrong_acts = activations[wrong_mask]

    min_n = min(len(correct_acts), len(wrong_acts))
    rng = np.random.default_rng(42)
    correct_acts = correct_acts[rng.permutation(len(correct_acts))[:min_n]]
    wrong_acts = wrong_acts[rng.permutation(len(wrong_acts))[:min_n]]
    split = int(min_n * 0.8)
    tc, tw = correct_acts[:split], wrong_acts[:split]

    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=1024, hidden_dims=[256, 64], activation="silu")
    ebm = GibbsModel(config, key=key)

    def get_p(m):
        return {"layers": [(w, b) for w, b in m.layers],
                "output_weight": m.output_weight, "output_bias": m.output_bias}

    def set_p(m, p):
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    params = get_p(ebm)
    def loss_fn(p):
        old = get_p(ebm); set_p(ebm, p)
        r = nce_loss(ebm, tc, tw); set_p(ebm, old); return r

    for ep in range(300):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.005 * g, params, grads)
    set_p(ebm, params)
    print("  EBM trained (300 epochs)")

    # --- Step 2: Load model ---
    print("\nStep 2: Loading Qwen3.5-0.8B...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        output_hidden_states=True,
        dtype=torch.float16 if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    # --- Step 3: Run on TruthfulQA subset ---
    print("\nStep 3: Running rejection sampling on TruthfulQA...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from collect_truthfulqa_activations import check_truthfulqa_answer

    ds = load_dataset("truthful_qa", "generation")
    questions = list(ds["validation"])[:100]  # First 100 for speed

    n_candidates = 5
    results = {"greedy": 0, "logprob": 0, "ebm": 0, "composite": 0}
    total = 0

    for qi, example in enumerate(questions):
        question = example["question"]
        correct_answers = example["correct_answers"]
        incorrect_answers = example["incorrect_answers"]
        best_answer = example.get("best_answer", "")

        prompt = f"Answer briefly and factually in one sentence. {question}"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        # Generate N candidates
        candidates = []
        for ci in range(n_candidates):
            gen_kwargs = {"max_new_tokens": 80, "pad_token_id": tokenizer.eos_token_id}
            if ci > 0:  # First is greedy, rest sampled
                gen_kwargs.update(do_sample=True, temperature=0.8, top_p=0.95)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, **gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            gen_ids = outputs.sequences[0, prompt_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()

            # Logprob
            total_lp = 0.0
            n_tok = 0
            if hasattr(outputs, "scores") and outputs.scores:
                for si, scores in enumerate(outputs.scores):
                    if si >= len(gen_ids):
                        break
                    tid = gen_ids[si].item()
                    lp = torch.log_softmax(scores[0], dim=-1)
                    total_lp += lp[tid].item()
                    n_tok += 1
            mean_lp = total_lp / max(n_tok, 1)

            # Hidden states for EBM
            with torch.no_grad():
                ho = model(outputs.sequences, output_hidden_states=True)
                hs = ho.hidden_states
            layer_hidden = hs[-1][0, prompt_len:, :].float().cpu().numpy()
            mean_ebm = score_activations_with_ebm(ebm, layer_hidden)

            is_correct = check_truthfulqa_answer(response, correct_answers, incorrect_answers, best_answer)

            candidates.append({
                "response": response,
                "logprob": mean_lp,
                "ebm": mean_ebm,
                "composite": 1.0 * mean_ebm - 1.0 * mean_lp,
                "correct": is_correct,
            })

        # Score each selection strategy
        total += 1

        # Greedy = first candidate
        if candidates[0]["correct"]:
            results["greedy"] += 1

        # Logprob-best = highest logprob
        lp_best = max(candidates, key=lambda c: c["logprob"])
        if lp_best["correct"]:
            results["logprob"] += 1

        # EBM-best = lowest EBM energy
        ebm_best = min(candidates, key=lambda c: c["ebm"])
        if ebm_best["correct"]:
            results["ebm"] += 1

        # Composite-best = lowest composite
        comp_best = min(candidates, key=lambda c: c["composite"])
        if comp_best["correct"]:
            results["composite"] += 1

        if (qi + 1) % 10 == 0:
            print(f"  [{qi+1:3d}/{len(questions)}] "
                  f"greedy={results['greedy']}/{total} "
                  f"logprob={results['logprob']}/{total} "
                  f"ebm={results['ebm']}/{total} "
                  f"composite={results['composite']}/{total}")

    # --- Results ---
    sep = "=" * 60
    print(f"\n{sep}")
    print("EXPERIMENT 23 RESULTS: EBM-Guided Rejection Sampling")
    print(sep)
    for method, count in results.items():
        acc = count / total * 100
        delta = acc - results["greedy"] / total * 100
        print(f"  {method:12s}: {count}/{total} ({acc:.1f}%) {delta:+.1f}%")
    print(sep)

    # Determine verdict
    greedy_acc = results["greedy"] / total
    composite_acc = results["composite"] / total
    if composite_acc > greedy_acc + 0.05:
        print("VERDICT: ✅ Composite EBM+logprob IMPROVES over greedy")
    elif composite_acc > greedy_acc:
        print("VERDICT: ⚠️ Small improvement")
    else:
        print("VERDICT: ❌ No improvement")

    return 0


if __name__ == "__main__":
    sys.exit(main())
