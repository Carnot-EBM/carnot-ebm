#!/usr/bin/env python3
"""Experiment 28: Multi-layer activation concatenation.

Experiment 24 showed hallucination signal follows a U-curve: present at
layers 4 and 24 but compressed in between. Concatenating early + late layer
activations may provide a richer feature set than either alone.

Tests on Qwen3.5-0.8B (200 TruthfulQA questions):
  - Last layer only (baseline): ~75% with no-thinking
  - Early + late concat: layers 4 + 24 → 2048-dim input
  - Multi-layer concat: layers 0, 4, 12, 24 → 4096-dim input
  - All sampled layers: every 4th → 7 * 1024 = 7168-dim input

Usage:
    sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        PYTHONUNBUFFERED=1 .venv/bin/python scripts/experiment_28_multilayer_concat.py'
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


def collect_multilayer_activations(
    n_questions: int = 200,
) -> tuple[dict[int, list], list, int]:
    """Collect per-token activations from ALL layers of Qwen3.5-0.8B.

    Returns (layer_acts, labels, hidden_dim) where layer_acts maps
    layer_index -> list of (n_tokens_per_question,) arrays.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    hidden_dim = 1024

    ds = load_dataset("truthful_qa", "generation")
    questions = list(ds["validation"])[:n_questions]

    # Collect from layers 0, 4, 8, 12, 16, 20, 24
    probe_layers = [0, 4, 8, 12, 16, 20, 24]
    layer_tokens: dict[int, list] = {l: [] for l in probe_layers}
    all_labels: list[int] = []
    n_correct = 0
    n_wrong = 0

    for qi, example in enumerate(questions):
        question = example["question"]
        correct_answers = example["correct_answers"]
        incorrect_answers = example["incorrect_answers"]
        best_answer = example.get("best_answer", "")

        messages = [{"role": "user", "content": f"Answer briefly and factually in one sentence. {question}"}]
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

        is_correct = check_truthfulqa_answer(response, correct_answers, incorrect_answers, best_answer)
        if is_correct:
            n_correct += 1
        else:
            n_wrong += 1

        # Get hidden states from all layers
        with torch.no_grad():
            ho = model(outputs, output_hidden_states=True)
            hs = ho.hidden_states

        n_gen = len(gen_ids)
        for layer_idx in probe_layers:
            if layer_idx < len(hs):
                layer_act = hs[layer_idx][0, prompt_len:, :].float().cpu().numpy()
                for t in range(n_gen):
                    layer_tokens[layer_idx].append(layer_act[t])

        for _ in range(n_gen):
            all_labels.append(1 if is_correct else 0)

        if (qi + 1) % 50 == 0:
            print(f"  [{qi+1:3d}/{n_questions}] correct={n_correct} wrong={n_wrong} "
                  f"({n_correct/(qi+1)*100:.0f}%) tokens={len(all_labels)}")

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return layer_tokens, all_labels, hidden_dim


def train_and_eval(activations: np.ndarray, labels: np.ndarray, input_dim: int, label: str):
    """Train EBM and evaluate. Returns (test_acc, gap)."""
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

    # Scale hidden dims with input size
    if input_dim <= 1024:
        hdims = [256, 64]
    elif input_dim <= 2048:
        hdims = [512, 128]
    elif input_dim <= 4096:
        hdims = [1024, 256]
    else:
        hdims = [1024, 256]  # Cap to avoid huge models

    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=input_dim, hidden_dims=hdims, activation="silu")
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
    print(f"  {label}: test={acc:.1%}, gap={gap:.4f}")
    return acc, gap


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 28: Multi-Layer Activation Concatenation")
    print("=" * 70)

    start = time.time()

    # Collect from all layers
    layer_tokens, all_labels, hidden_dim = collect_multilayer_activations(200)
    labels = np.array(all_labels, dtype=np.int32)
    n_tokens = len(labels)
    print(f"\nCollected {n_tokens} tokens from 7 layers")

    # Convert to arrays
    layer_arrays = {}
    for l_idx, tokens in layer_tokens.items():
        layer_arrays[l_idx] = np.array(tokens, dtype=np.float32)

    print(f"\n--- Testing Different Layer Combinations ---\n")

    results = {}

    # 1. Last layer only (baseline)
    last = layer_arrays[24]
    acc, gap = train_and_eval(last, labels, hidden_dim, "Layer 24 only (baseline)")
    results["last_only"] = acc

    # 2. Early layer only (layer 4)
    early = layer_arrays[4]
    acc, gap = train_and_eval(early, labels, hidden_dim, "Layer 4 only")
    results["early_only"] = acc

    # 3. Early + Late concat (layers 4 + 24 → 2048-dim)
    early_late = np.concatenate([layer_arrays[4], layer_arrays[24]], axis=1)
    acc, gap = train_and_eval(early_late, labels, hidden_dim * 2, "Layers 4+24 concat (2048-dim)")
    results["early_late"] = acc

    # 4. Three layers (4 + 12 + 24 → 3072-dim)
    three = np.concatenate([layer_arrays[4], layer_arrays[12], layer_arrays[24]], axis=1)
    acc, gap = train_and_eval(three, labels, hidden_dim * 3, "Layers 4+12+24 concat (3072-dim)")
    results["three_layers"] = acc

    # 5. Four layers (0 + 4 + 12 + 24 → 4096-dim)
    four = np.concatenate([layer_arrays[0], layer_arrays[4], layer_arrays[12], layer_arrays[24]], axis=1)
    acc, gap = train_and_eval(four, labels, hidden_dim * 4, "Layers 0+4+12+24 concat (4096-dim)")
    results["four_layers"] = acc

    # 6. All sampled layers (0,4,8,12,16,20,24 → 7168-dim)
    all_concat = np.concatenate([layer_arrays[l] for l in sorted(layer_arrays.keys())], axis=1)
    acc, gap = train_and_eval(all_concat, labels, hidden_dim * 7, "All 7 layers concat (7168-dim)")
    results["all_layers"] = acc

    elapsed = time.time() - start

    # Summary
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 28 RESULTS ({elapsed:.0f}s)")
    print(sep)
    print(f"  Model: Qwen3.5-0.8B (no thinking), {n_tokens} tokens")
    print(f"")
    for name, acc in results.items():
        delta = acc - results["last_only"]
        marker = " ← baseline" if name == "last_only" else f" ({delta:+.1%})"
        print(f"  {name:25s}: {acc:.1%}{marker}")

    best_name = max(results, key=results.get)
    best_acc = results[best_name]
    baseline = results["last_only"]

    print(f"\n  Best: {best_name} ({best_acc:.1%})")
    if best_acc > baseline + 0.03:
        print(f"  VERDICT: ✅ Multi-layer concatenation improves by {best_acc - baseline:.1%}")
    elif best_acc > baseline:
        print(f"  VERDICT: ⚠️ Small improvement ({best_acc - baseline:+.1%})")
    else:
        print(f"  VERDICT: ❌ No improvement over single last layer")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
