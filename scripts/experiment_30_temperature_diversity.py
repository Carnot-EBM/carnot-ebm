#!/usr/bin/env python3
"""Experiment 30: Temperature-diverse training data for EBM detection.

Previous experiments used greedy decoding (temp=0) — one deterministic answer
per question. This experiment generates N responses per question at varying
temperatures, labels each independently, and trains the EBM on the larger
and more diverse activation dataset.

Tests:
  1. Greedy-only baseline (200 questions, ~7K tokens)
  2. Multi-temp (5 samples/question at temp=0.3,0.5,0.7,0.9,1.0 → ~35K tokens)
  3. Cross-temp transfer: train on greedy, test on sampled (and vice versa)

Model: Qwen3.5-0.8B (fastest, well-characterized)

Usage:
    sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        PYTHONUNBUFFERED=1 .venv/bin/python scripts/experiment_30_temperature_diversity.py'
"""

from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from collect_truthfulqa_activations import check_truthfulqa_answer


def collect_with_temperatures(
    n_questions: int = 200,
    temperatures: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect per-token activations at multiple temperatures.

    For each question, generates one response per temperature.
    Returns (activations, labels, temp_ids, question_ids).
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if temperatures is None:
        temperatures = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3.5-0.8B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        output_hidden_states=True,
        torch_dtype=torch.float16 if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    ds = load_dataset("truthful_qa", "generation")
    questions = list(ds["validation"])[:n_questions]

    all_activations = []
    all_labels = []
    all_temps = []
    all_qids = []
    stats = {t: {"correct": 0, "wrong": 0} for t in temperatures}

    for qi, example in enumerate(questions):
        question = example["question"]
        correct_answers = example["correct_answers"]
        incorrect_answers = example["incorrect_answers"]
        best_answer = example.get("best_answer", "")

        prompt = f"Answer briefly and factually in one sentence. {question}"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        for temp in temperatures:
            gen_kwargs = {
                "max_new_tokens": 80,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if temp == 0.0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temp
                gen_kwargs["top_p"] = 0.95

            try:
                with torch.no_grad():
                    outputs = model.generate(**inputs, **gen_kwargs)

                gen_ids = outputs[0, prompt_len:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True)
                if "</think>" in response:
                    response = response.split("</think>")[-1].strip()

                is_correct = check_truthfulqa_answer(
                    response, correct_answers, incorrect_answers, best_answer,
                )

                # Get hidden states
                with torch.no_grad():
                    ho = model(outputs, output_hidden_states=True)
                    hs = ho.hidden_states

                last_layer = hs[-1][0, prompt_len:, :].float().cpu().numpy()

                for t in range(len(gen_ids)):
                    all_activations.append(last_layer[t])
                    all_labels.append(1 if is_correct else 0)
                    all_temps.append(temp)
                    all_qids.append(qi)

                if is_correct:
                    stats[temp]["correct"] += 1
                else:
                    stats[temp]["wrong"] += 1

            except Exception as e:
                stats[temp]["wrong"] += 1

        if (qi + 1) % 50 == 0:
            n_tok = len(all_activations)
            print(f"  [{qi+1:3d}/{n_questions}] tokens={n_tok}")
            for t in temperatures:
                c, w = stats[t]["correct"], stats[t]["wrong"]
                total = c + w
                acc = c / total * 100 if total > 0 else 0
                print(f"    temp={t:.1f}: {c}/{total} ({acc:.0f}%)")

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return (
        np.array(all_activations, dtype=np.float32),
        np.array(all_labels, dtype=np.int32),
        np.array(all_temps, dtype=np.float32),
        np.array(all_qids, dtype=np.int32),
    )


def train_and_eval(activations, labels, hidden_dim, label):
    """Train EBM and evaluate."""
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    acts = jnp.array(activations)
    correct = acts[labels == 1]
    wrong = acts[labels == 0]
    min_n = min(len(correct), len(wrong))
    if min_n < 20:
        print(f"  {label}: SKIP — {min_n} samples")
        return 0.5, 0.0

    rng = np.random.default_rng(42)
    correct = correct[rng.permutation(len(correct))[:min_n]]
    wrong = wrong[rng.permutation(len(wrong))[:min_n]]
    split = int(min_n * 0.8)
    tc, tw = correct[:split], wrong[:split]
    vc, vw = correct[split:], wrong[split:]

    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=hidden_dim, hidden_dims=[256, 64], activation="silu")
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
        old = get_p(ebm)
        set_p(ebm, p)
        r = nce_loss(ebm, tc, tw)
        set_p(ebm, old)
        return r

    for _ in range(300):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.005 * g, params, grads)
    set_p(ebm, params)

    n_eval = min(300, len(vc))
    ce = [float(ebm.energy(vc[i])) for i in range(n_eval)]
    we = [float(ebm.energy(vw[i])) for i in range(n_eval)]
    thresh = (np.mean(ce) + np.mean(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    acc = (tp + tn) / (len(ce) + len(we))
    gap = np.mean(we) - np.mean(ce)

    print(f"  {label}: test={acc:.1%}, gap={gap:.4f}, n_train={split}, n_test={min_n-split}")
    return acc, gap


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 30: Temperature-Diverse Training Data")
    print("  Model: Qwen3.5-0.8B, 200 TruthfulQA questions")
    print("  Temperatures: 0.0 (greedy), 0.3, 0.5, 0.7, 0.9, 1.0")
    print("=" * 70)

    start = time.time()
    hidden_dim = 1024

    # Collect activations at all temperatures
    activations, labels, temps, qids = collect_with_temperatures(200)
    n_total = len(labels)
    print(f"\nTotal tokens collected: {n_total}")
    print(f"  Correct: {int(labels.sum())}, Wrong: {n_total - int(labels.sum())}")

    # --- Test 1: Greedy-only baseline ---
    print(f"\n--- Test 1: Greedy Only (temp=0.0) ---")
    greedy_mask = temps == 0.0
    a1_acc, a1_gap = train_and_eval(
        activations[greedy_mask], labels[greedy_mask], hidden_dim, "Greedy only")

    # --- Test 2: All temperatures combined ---
    print(f"\n--- Test 2: All Temperatures Combined ---")
    a2_acc, a2_gap = train_and_eval(activations, labels, hidden_dim, "All temps combined")

    # --- Test 3: Sampled only (no greedy) ---
    print(f"\n--- Test 3: Sampled Only (temp > 0) ---")
    sampled_mask = temps > 0.0
    a3_acc, a3_gap = train_and_eval(
        activations[sampled_mask], labels[sampled_mask], hidden_dim, "Sampled only")

    # --- Test 4: Per-temperature breakdown ---
    print(f"\n--- Test 4: Per-Temperature Breakdown ---")
    temp_results = {}
    for t in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
        mask = temps == t
        n_tok = int(mask.sum())
        n_correct = int(labels[mask].sum())
        acc, gap = train_and_eval(
            activations[mask], labels[mask], hidden_dim, f"temp={t:.1f}")
        temp_results[t] = {"acc": acc, "gap": gap, "n_tokens": n_tok, "n_correct": n_correct}

    # --- Test 5: Cross-temperature transfer ---
    print(f"\n--- Test 5: Cross-Temperature Transfer ---")
    # Train on greedy, test on sampled
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    greedy_acts = jnp.array(activations[greedy_mask])
    greedy_labs = labels[greedy_mask]
    sampled_acts = jnp.array(activations[sampled_mask])
    sampled_labs = labels[sampled_mask]

    # Train on greedy
    gc_acts = greedy_acts[greedy_labs == 1]
    gw_acts = greedy_acts[greedy_labs == 0]
    min_g = min(len(gc_acts), len(gw_acts))
    rng = np.random.default_rng(42)
    gc_acts = gc_acts[rng.permutation(len(gc_acts))[:min_g]]
    gw_acts = gw_acts[rng.permutation(len(gw_acts))[:min_g]]
    split_g = int(min_g * 0.8)

    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=hidden_dim, hidden_dims=[256, 64], activation="silu")
    ebm_greedy = GibbsModel(config, key=key)

    def get_p(m):
        return {"layers": [(w, b) for w, b in m.layers],
                "output_weight": m.output_weight, "output_bias": m.output_bias}

    def set_p(m, p):
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    params = get_p(ebm_greedy)

    def loss_fn(p):
        old = get_p(ebm_greedy)
        set_p(ebm_greedy, p)
        r = nce_loss(ebm_greedy, gc_acts[:split_g], gw_acts[:split_g])
        set_p(ebm_greedy, old)
        return r

    for _ in range(300):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.005 * g, params, grads)
    set_p(ebm_greedy, params)

    # Evaluate greedy-trained EBM on sampled data
    sc_acts = sampled_acts[sampled_labs == 1]
    sw_acts = sampled_acts[sampled_labs == 0]
    min_s = min(len(sc_acts), len(sw_acts), 300)
    sc_eval = sc_acts[rng.permutation(len(sc_acts))[:min_s]]
    sw_eval = sw_acts[rng.permutation(len(sw_acts))[:min_s]]

    ce = [float(ebm_greedy.energy(sc_eval[i])) for i in range(min_s)]
    we = [float(ebm_greedy.energy(sw_eval[i])) for i in range(min_s)]
    thresh = (np.mean(ce) + np.mean(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    cross_acc = (tp + tn) / (len(ce) + len(we))
    print(f"  Train greedy → test sampled: {cross_acc:.1%}")

    # Summary
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 30 RESULTS ({elapsed:.0f}s)")
    print(sep)
    print(f"  Total tokens: {n_total} ({n_total // 200} per question avg)")
    print(f"")
    print(f"  {'Dataset':25s} {'Accuracy':>9s} {'Gap':>8s} {'Tokens':>8s}")
    print(f"  {'-'*55}")
    print(f"  {'Greedy only (baseline)':25s} {a1_acc:>8.1%} {a1_gap:>8.4f} {int(greedy_mask.sum()):>8d}")
    print(f"  {'All temps combined':25s} {a2_acc:>8.1%} {a2_gap:>8.4f} {n_total:>8d}")
    print(f"  {'Sampled only (temp>0)':25s} {a3_acc:>8.1%} {a3_gap:>8.4f} {int(sampled_mask.sum()):>8d}")
    print(f"  {'Train greedy→test sampled':25s} {cross_acc:>8.1%}")
    print(f"")
    print(f"  Per-temperature:")
    for t, r in sorted(temp_results.items()):
        model_acc = r["n_correct"] / (r["n_tokens"] / 35) if r["n_tokens"] > 0 else 0  # approx
        print(f"    temp={t:.1f}: EBM={r['acc']:.1%}, gap={r['gap']:.4f}, tokens={r['n_tokens']}")

    delta = a2_acc - a1_acc
    if delta > 0.03:
        print(f"\n  VERDICT: ✅ Temperature diversity improves by {delta:.1%}")
    elif delta > 0:
        print(f"\n  VERDICT: ⚠️ Small improvement ({delta:+.1%})")
    else:
        print(f"\n  VERDICT: ❌ No improvement from temperature diversity ({delta:+.1%})")

    if cross_acc > 0.65:
        print(f"  Cross-temp transfer: ✅ Greedy-trained EBM works on sampled ({cross_acc:.1%})")
    else:
        print(f"  Cross-temp transfer: ❌ Greedy→sampled doesn't transfer well ({cross_acc:.1%})")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
