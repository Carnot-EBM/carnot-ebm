#!/usr/bin/env python3
"""Experiment 38: NLI-based EBM — use a factual consistency encoder.

Experiment 37 failed because sentence encoders embed topic similarity,
not factual correctness. NLI (Natural Language Inference) models ARE
trained to distinguish "supports" from "contradicts" — the closest
existing representation to factual correctness.

Approach:
  1. Use DeBERTa-v3 NLI cross-encoder to get per-token hidden states
     for "Question: X Answer: Y" inputs
  2. Extract [CLS] token or mean-pooled representation (768-dim)
  3. Train Gibbs EBM on these NLI-informed representations
  4. Test: does the NLI representation separate correct from wrong?
  5. If yes: attempt gradient-based repair in NLI embedding space

Also test: does the NLI model's own classification logit already
separate correct from wrong? (If so, we don't need the EBM at all.)

Usage:
    .venv/bin/python scripts/experiment_38_nli_ebm.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def main() -> int:
    import torch
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    print("=" * 70)
    print("EXPERIMENT 38: NLI-Based EBM for Factual Correctness")
    print("=" * 70)

    start = time.time()

    # --- Step 1: Load NLI model ---
    print("\nStep 1: Loading NLI cross-encoder...")
    model_name = "cross-encoder/nli-deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, output_hidden_states=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nli_model = nli_model.to(device)
    nli_model.eval()
    print(f"  Model: {model_name} (768-dim, 6 layers)")
    print(f"  Device: {device}")

    # --- Step 2: Embed TruthfulQA with NLI model ---
    print("\nStep 2: Encoding QA pairs through NLI model...")

    ds = load_dataset("truthful_qa", "generation")
    examples = list(ds["validation"])

    correct_hiddens = []
    wrong_hiddens = []
    correct_logits = []
    wrong_logits = []

    for qi, ex in enumerate(examples):
        question = ex["question"]

        for answer in ex["correct_answers"]:
            # NLI format: premise [SEP] hypothesis
            # We frame it as: "The answer to '{question}' is factually correct." [SEP] answer
            inputs = tokenizer(
                f"The answer to the question '{question}' is factually correct.",
                answer,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs = nli_model(**inputs)

            # Get CLS hidden state from last layer
            hidden = outputs.hidden_states[-1][0, 0, :].float().cpu().numpy()
            correct_hiddens.append(hidden)
            # Get NLI logits (entailment, neutral, contradiction)
            correct_logits.append(outputs.logits[0].float().cpu().numpy())

        for answer in ex["incorrect_answers"]:
            inputs = tokenizer(
                f"The answer to the question '{question}' is factually correct.",
                answer,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs = nli_model(**inputs)

            hidden = outputs.hidden_states[-1][0, 0, :].float().cpu().numpy()
            wrong_hiddens.append(hidden)
            wrong_logits.append(outputs.logits[0].float().cpu().numpy())

        if (qi + 1) % 100 == 0:
            print(f"    [{qi+1}/{len(examples)}] correct={len(correct_hiddens)} wrong={len(wrong_hiddens)}")

    correct_hiddens = np.array(correct_hiddens, dtype=np.float32)
    wrong_hiddens = np.array(wrong_hiddens, dtype=np.float32)
    correct_logits = np.array(correct_logits, dtype=np.float32)
    wrong_logits = np.array(wrong_logits, dtype=np.float32)

    print(f"  Correct: {len(correct_hiddens)}, Wrong: {len(wrong_hiddens)}")

    # --- Step 3: Check NLI logits directly ---
    print("\nStep 3: Does NLI classification already separate correct/wrong?")

    # NLI labels: 0=entailment, 1=neutral, 2=contradiction (model-dependent)
    # Check which logit dimension best separates
    for dim, label in enumerate(["entailment", "neutral", "contradiction"]):
        c_mean = correct_logits[:, dim].mean()
        w_mean = wrong_logits[:, dim].mean()
        thresh = (c_mean + w_mean) / 2
        if c_mean > w_mean:
            tp = sum(1 for l in wrong_logits[:, dim] if l < thresh)
            tn = sum(1 for l in correct_logits[:, dim] if l >= thresh)
        else:
            tp = sum(1 for l in wrong_logits[:, dim] if l > thresh)
            tn = sum(1 for l in correct_logits[:, dim] if l <= thresh)
        acc = (tp + tn) / (len(correct_logits) + len(wrong_logits))
        print(f"  {label:15s}: correct={c_mean:+.3f} wrong={w_mean:+.3f} acc={acc:.1%}")

    # --- Step 4: Check hidden state separation ---
    print("\nStep 4: Hidden state centroid separation...")
    c_centroid = correct_hiddens.mean(axis=0)
    w_centroid = wrong_hiddens.mean(axis=0)
    cos = np.dot(c_centroid, w_centroid) / (np.linalg.norm(c_centroid) * np.linalg.norm(w_centroid))
    l2 = np.linalg.norm(c_centroid - w_centroid)
    print(f"  Centroid cosine similarity: {cos:.4f}")
    print(f"  Centroid L2 distance: {l2:.4f}")

    # --- Step 5: Train Gibbs EBM on NLI hidden states ---
    print("\nStep 5: Training Gibbs EBM on NLI hidden states...")

    min_n = min(len(correct_hiddens), len(wrong_hiddens))
    rng = np.random.default_rng(42)
    c_shuf = jnp.array(correct_hiddens[rng.permutation(len(correct_hiddens))[:min_n]])
    w_shuf = jnp.array(wrong_hiddens[rng.permutation(len(wrong_hiddens))[:min_n]])
    split = int(min_n * 0.8)
    tc, tw = c_shuf[:split], w_shuf[:split]
    vc, vw = c_shuf[split:], w_shuf[split:]

    print(f"  Balanced: {min_n} each (train={split}, test={min_n - split})")

    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=768, hidden_dims=[256, 64], activation="silu")
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

    for ep in range(500):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.005 * g, params, grads)
        if ep % 100 == 0:
            set_p(ebm, params)
            loss = float(nce_loss(ebm, tc, tw))
            print(f"    Epoch {ep}: loss={loss:.4f}")
    set_p(ebm, params)

    # Evaluate
    n_eval = min(300, len(vc))
    ce = [float(ebm.energy(vc[i])) for i in range(n_eval)]
    we = [float(ebm.energy(vw[i])) for i in range(n_eval)]
    thresh = (np.mean(ce) + np.mean(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    ebm_acc = (tp + tn) / (len(ce) + len(we))
    gap = np.mean(we) - np.mean(ce)

    # --- Results ---
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 38 RESULTS ({elapsed:.0f}s)")
    print(sep)
    print(f"  NLI model: {model_name}")
    print(f"  Centroid cosine: {cos:.4f} (cf. sentence encoder: 0.970)")
    print(f"  Centroid L2: {l2:.4f} (cf. sentence encoder: 0.091)")
    print(f"")
    print(f"  EBM on NLI hidden states:")
    print(f"    Test accuracy: {ebm_acc:.1%}")
    print(f"    Energy gap: {gap:.4f}")
    print(f"")
    print(f"  Comparison:")
    print(f"    Sentence encoder EBM (exp 37):  57.5% (loss never decreased)")
    print(f"    LLM activation EBM (exp 1-36):  75.5% test / 50% practical")
    print(f"    NLI hidden state EBM (this):    {ebm_acc:.1%}")

    if ebm_acc > 0.75:
        print(f"\n  VERDICT: ✅ NLI representations enable factual discrimination!")
        print(f"  The NLI encoder provides a representation where correct and wrong")
        print(f"  answers are separable — unlike sentence encoders or LLM activations.")
    elif ebm_acc > 0.6:
        print(f"\n  VERDICT: ⚠️ NLI provides some signal ({ebm_acc:.1%}) but not strong")
    else:
        print(f"\n  VERDICT: ❌ NLI hidden states don't separate correct from wrong")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
