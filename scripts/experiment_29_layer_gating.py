#!/usr/bin/env python3
"""Experiment 29: Learned layer gating vs fixed concatenation for EBM detection.

Compare three approaches to multi-layer activation aggregation:

1. **Last layer only** — baseline (1x hidden_dim)
2. **All layers concat** — concatenate all 25 layers (25x hidden_dim)
3. **Learned gating** — one learned scalar weight per layer, weighted sum → hidden_dim
4. **Learned attention** — per-token attention over layers → hidden_dim

Tests on Qwen3.5-0.8B (24 layers, 1024-dim, fast) with 200 TruthfulQA questions.
Also measures EBM size and inference speed for each approach.

Usage:
    sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        PYTHONUNBUFFERED=1 .venv/bin/python scripts/experiment_29_layer_gating.py'
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


def collect_all_layers(n_questions: int = 200) -> tuple[dict[int, np.ndarray], np.ndarray, int]:
    """Collect per-token activations from ALL layers of Qwen3.5-0.8B.

    Returns (layer_arrays, labels, hidden_dim).
    layer_arrays maps layer_index (0..24) to np.array of shape (n_tokens, 1024).
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
    n_layers = 25  # 24 transformer layers + 1 embedding = 25 hidden states

    ds = load_dataset("truthful_qa", "generation")
    questions = list(ds["validation"])[:n_questions]

    # Collect from ALL layers
    layer_tokens: dict[int, list] = {l: [] for l in range(n_layers)}
    all_labels: list[int] = []
    n_correct = 0
    n_wrong = 0

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

        with torch.no_grad():
            ho = model(outputs, output_hidden_states=True)
            hs = ho.hidden_states

        n_gen = len(gen_ids)
        for layer_idx in range(min(n_layers, len(hs))):
            act = hs[layer_idx][0, prompt_len:, :].float().cpu().numpy()
            for t in range(n_gen):
                layer_tokens[layer_idx].append(act[t])

        for _ in range(n_gen):
            all_labels.append(1 if is_correct else 0)

        if (qi + 1) % 50 == 0:
            print(f"  [{qi+1:3d}/{n_questions}] correct={n_correct} wrong={n_wrong} "
                  f"({n_correct/(qi+1)*100:.0f}%) tokens={len(all_labels)}")

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    layer_arrays = {l: np.array(tokens, dtype=np.float32) for l, tokens in layer_tokens.items() if tokens}
    labels = np.array(all_labels, dtype=np.int32)
    print(f"  Collected: {len(labels)} tokens from {len(layer_arrays)} layers, "
          f"{n_correct} correct, {n_wrong} wrong")

    return layer_arrays, labels, hidden_dim


def train_ebm_and_eval(activations: np.ndarray, labels: np.ndarray, input_dim: int,
                       label: str, hidden_dims: list[int] | None = None):
    """Train EBM, evaluate, measure size and speed."""
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
        return 0.5, 0.0, 0, 0.0

    rng = np.random.default_rng(42)
    correct = correct[rng.permutation(len(correct))[:min_n]]
    wrong = wrong[rng.permutation(len(wrong))[:min_n]]
    split = int(min_n * 0.8)
    tc, tw = correct[:split], wrong[:split]
    vc, vw = correct[split:], wrong[split:]

    if hidden_dims is None:
        if input_dim <= 1024:
            hidden_dims = [256, 64]
        elif input_dim <= 4096:
            hidden_dims = [512, 128]
        else:
            hidden_dims = [1024, 256]

    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=input_dim, hidden_dims=hidden_dims, activation="silu")
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

    train_start = time.time()
    for _ in range(300):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.005 * g, params, grads)
    set_p(ebm, params)
    train_time = time.time() - train_start

    # Count parameters
    n_params = sum(w.size + b.size for w, b in ebm.layers) + ebm.output_weight.size + ebm.output_bias.size

    # Evaluate
    n_eval = min(300, len(vc))
    infer_start = time.time()
    ce = [float(ebm.energy(vc[i])) for i in range(n_eval)]
    we = [float(ebm.energy(vw[i])) for i in range(n_eval)]
    infer_time = (time.time() - infer_start) / (2 * n_eval) * 1000  # ms per token

    thresh = (np.mean(ce) + np.mean(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    acc = (tp + tn) / (len(ce) + len(we))
    gap = np.mean(we) - np.mean(ce)

    print(f"  {label}: test={acc:.1%}, gap={gap:.4f}, params={n_params:,}, "
          f"train={train_time:.1f}s, infer={infer_time:.2f}ms/tok")
    return acc, gap, n_params, infer_time


def learned_gating(layer_arrays: dict[int, np.ndarray], labels: np.ndarray, hidden_dim: int):
    """Learn per-layer scalar weights via gradient descent, then train EBM on weighted sum.

    The gating learns which layers matter most. Output is still hidden_dim dimensional.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    n_layers = len(layer_arrays)
    # Stack all layers: (n_tokens, n_layers, hidden_dim)
    stacked = np.stack([layer_arrays[l] for l in sorted(layer_arrays.keys())], axis=1)
    stacked = jnp.array(stacked)

    correct_mask = labels == 1
    wrong_mask = labels == 0
    correct_stacked = stacked[correct_mask]
    wrong_stacked = stacked[wrong_mask]
    min_n = min(len(correct_stacked), len(wrong_stacked))

    rng = np.random.default_rng(42)
    correct_stacked = correct_stacked[rng.permutation(len(correct_stacked))[:min_n]]
    wrong_stacked = wrong_stacked[rng.permutation(len(wrong_stacked))[:min_n]]
    split = int(min_n * 0.8)
    tc_s, tw_s = correct_stacked[:split], wrong_stacked[:split]
    vc_s, vw_s = correct_stacked[split:], wrong_stacked[split:]

    # Initialize gate weights (uniform)
    key = jrandom.PRNGKey(42)
    gate_logits = jnp.zeros(n_layers)

    # EBM on gated output
    config = GibbsConfig(input_dim=hidden_dim, hidden_dims=[256, 64], activation="silu")
    ebm = GibbsModel(config, key=key)

    def get_p(m):
        return {"layers": [(w, b) for w, b in m.layers],
                "output_weight": m.output_weight, "output_bias": m.output_bias}

    def set_p(m, p):
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    def apply_gate(stacked_batch, gate_logits):
        """Apply softmax gating to layers: weighted sum → (batch, hidden_dim)."""
        weights = jax.nn.softmax(gate_logits)  # (n_layers,)
        return jnp.einsum("btd,t->bd", stacked_batch, weights)

    params = get_p(ebm)

    def loss_fn(params_and_gate):
        p, gl = params_and_gate
        old = get_p(ebm)
        set_p(ebm, p)
        tc_gated = apply_gate(tc_s, gl)
        tw_gated = apply_gate(tw_s, gl)
        r = nce_loss(ebm, tc_gated, tw_gated)
        set_p(ebm, old)
        return r

    combined = (params, gate_logits)

    train_start = time.time()
    for ep in range(300):
        grads = jax.grad(loss_fn)(combined)
        combined = jax.tree.map(lambda p, g: p - 0.005 * g, combined, grads)
    params, gate_logits = combined
    set_p(ebm, params)
    train_time = time.time() - train_start

    # Show learned gate weights
    gate_weights = jax.nn.softmax(gate_logits)
    top_layers = sorted(range(n_layers), key=lambda i: float(gate_weights[i]), reverse=True)[:5]
    print(f"  Top 5 gate weights: {', '.join(f'L{l}={float(gate_weights[l]):.3f}' for l in top_layers)}")

    # Evaluate
    n_eval = min(300, len(vc_s))
    vc_gated = apply_gate(vc_s[:n_eval], gate_logits)
    vw_gated = apply_gate(vw_s[:n_eval], gate_logits)

    infer_start = time.time()
    ce = [float(ebm.energy(vc_gated[i])) for i in range(n_eval)]
    we = [float(ebm.energy(vw_gated[i])) for i in range(n_eval)]
    infer_time = (time.time() - infer_start) / (2 * n_eval) * 1000

    thresh = (np.mean(ce) + np.mean(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    acc = (tp + tn) / (len(ce) + len(we))
    gap = np.mean(we) - np.mean(ce)

    n_params = sum(w.size + b.size for w, b in ebm.layers) + ebm.output_weight.size + ebm.output_bias.size + n_layers

    print(f"  Learned gating: test={acc:.1%}, gap={gap:.4f}, params={n_params:,}, "
          f"train={train_time:.1f}s, infer={infer_time:.2f}ms/tok")
    return acc, gap, n_params, infer_time, gate_weights


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 29: Learned Layer Gating vs Fixed Concatenation")
    print("  Model: Qwen3.5-0.8B (24 layers, 1024-dim)")
    print("=" * 70)

    start = time.time()

    # Collect all layers
    layer_arrays, labels, hidden_dim = collect_all_layers(200)
    n_layers = len(layer_arrays)
    n_tokens = len(labels)

    print(f"\n--- Approach 1: Last Layer Only (baseline) ---")
    last = layer_arrays[n_layers - 1]
    a1_acc, a1_gap, a1_params, a1_speed = train_ebm_and_eval(
        last, labels, hidden_dim, "Last layer (L24)")

    print(f"\n--- Approach 2: All Layers Concatenated ---")
    all_concat = np.concatenate([layer_arrays[l] for l in sorted(layer_arrays.keys())], axis=1)
    concat_dim = hidden_dim * n_layers
    print(f"  Concat dim: {concat_dim}")
    a2_acc, a2_gap, a2_params, a2_speed = train_ebm_and_eval(
        all_concat, labels, concat_dim, "All layers concat",
        hidden_dims=[1024, 256])

    print(f"\n--- Approach 3: Best-3 Layers Concat (early+mid+late) ---")
    # layers 4, 12, 24 (or equivalent fractions of 25 layers)
    early, mid, late = 4, 12, n_layers - 1
    three_concat = np.concatenate([layer_arrays[early], layer_arrays[mid], layer_arrays[late]], axis=1)
    a3_acc, a3_gap, a3_params, a3_speed = train_ebm_and_eval(
        three_concat, labels, hidden_dim * 3, "3-layer concat (4+12+24)",
        hidden_dims=[512, 128])

    print(f"\n--- Approach 4: Learned Layer Gating ---")
    a4_acc, a4_gap, a4_params, a4_speed, gate_weights = learned_gating(
        layer_arrays, labels, hidden_dim)

    # Summary
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 29 RESULTS ({elapsed:.0f}s)")
    print(f"  Model: Qwen3.5-0.8B, {n_tokens} tokens, {n_layers} layers")
    print(sep)
    print(f"{'Approach':30s} {'Accuracy':>9s} {'Params':>10s} {'ms/tok':>8s} {'Input Dim':>10s}")
    print("-" * 70)

    results = [
        ("Last layer only", a1_acc, a1_params, a1_speed, hidden_dim),
        ("All layers concat", a2_acc, a2_params, a2_speed, concat_dim),
        ("3-layer concat (4+12+24)", a3_acc, a3_params, a3_speed, hidden_dim * 3),
        ("Learned gating", a4_acc, a4_params, a4_speed, hidden_dim),
    ]

    best_name, best_acc = "", 0
    for name, acc, params, speed, dim in results:
        marker = " *" if acc == max(r[1] for r in results) else ""
        print(f"  {name:28s} {acc:>8.1%} {params:>10,} {speed:>7.2f} {dim:>10,}{marker}")
        if acc > best_acc:
            best_acc = acc
            best_name = name

    print(sep)
    print(f"  Best: {best_name} ({best_acc:.1%})")

    baseline = a1_acc
    if best_acc > baseline + 0.03:
        print(f"  VERDICT: ✅ {best_name} improves by {best_acc - baseline:.1%} over baseline")
    elif best_acc > baseline:
        print(f"  VERDICT: ⚠️ Small improvement ({best_acc - baseline:+.1%})")
    else:
        print(f"  VERDICT: ❌ No improvement over last-layer baseline")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
