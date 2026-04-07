#!/usr/bin/env python3
"""Experiment 27: Upstream hallucination detection — predict from the QUESTION.

Principle 9 says post-hoc answer analysis fails on adversarial questions.
Can we detect adversarial questions BEFORE generation by analyzing the
model's representations of the question itself?

Approach:
  1. Feed each TruthfulQA question through the model (no generation).
  2. Extract the last token's hidden state (the model's "understanding" of the question).
  3. Label: did the model actually get this question right? (from prior data)
  4. Train EBM on question representations to predict future hallucination.

If this works, we can flag "this question is dangerous" before wasting
compute on generation — and apply extra scrutiny (more candidates,
structural verification, or abstaining).

Tests on: LFM2.5-350M (fastest), Qwen3.5-0.8B, Gemma4-E2B-it.

Usage:
    .venv/bin/python scripts/experiment_27_upstream_detection.py
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def collect_question_activations(
    model_name: str,
    short_id: str,
    has_chat: bool,
    n_questions: int = 400,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Extract question-only activations (no generation needed).

    For each question:
    1. Tokenize the prompt (question only).
    2. Forward pass to get hidden states.
    3. Take the LAST token's hidden state as the question representation.
    4. Also generate an answer to determine the label (correct/wrong).

    Returns (question_activations, labels, hidden_dim, n_correct, n_wrong).
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        output_hidden_states=True,
        torch_dtype=torch.float16 if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    config = model.config
    if hasattr(config, 'get_text_config'):
        tc = config.get_text_config()
    else:
        tc = config
    hidden_dim = getattr(tc, 'hidden_size', getattr(tc, 'd_model', 1024))
    print(f"  Hidden dim: {hidden_dim}, Device: {device}")

    ds = load_dataset("truthful_qa", "generation")
    questions = list(ds["validation"])[:n_questions]

    question_acts = []  # One activation per question (last token of prompt)
    labels = []
    n_correct = 0
    n_wrong = 0

    for qi, example in enumerate(questions):
        question = example["question"]
        correct_answers = example["correct_answers"]
        incorrect_answers = example["incorrect_answers"]
        best_answer = example.get("best_answer", "")

        prompt = f"Answer briefly and factually in one sentence. {question}"

        if has_chat and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        # Step 1: Get question-only activations (just a forward pass, no generation)
        with torch.no_grad():
            prompt_out = model(**inputs, output_hidden_states=True)
            hs = prompt_out.hidden_states

        # Last layer, last token = model's "understanding" of the question
        last_token_act = hs[-1][0, -1, :].float().cpu().numpy()
        # Also try mean-pooling all question tokens
        mean_act = hs[-1][0, :, :].float().mean(dim=0).cpu().numpy()

        # Step 2: Generate answer to get the label
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=80, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_ids = outputs[0, prompt_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()

            is_correct = check_truthfulqa_answer(
                response, correct_answers, incorrect_answers, best_answer,
            )
        except Exception:
            is_correct = False

        if is_correct:
            n_correct += 1
        else:
            n_wrong += 1

        # Use last-token representation (the model's compressed understanding)
        question_acts.append(last_token_act)
        labels.append(1 if is_correct else 0)

        if (qi + 1) % 100 == 0:
            print(f"    [{qi+1:3d}/{n_questions}] correct={n_correct} wrong={n_wrong} "
                  f"({n_correct/(qi+1)*100:.0f}%)")

    # Free GPU
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return np.array(question_acts, dtype=np.float32), np.array(labels, dtype=np.int32), hidden_dim, n_correct, n_wrong


def train_and_evaluate(activations: np.ndarray, labels: np.ndarray, hidden_dim: int, label: str):
    """Train EBM on question activations and evaluate."""
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    acts = jnp.array(activations)
    correct = acts[labels == 1]
    wrong = acts[labels == 0]
    min_n = min(len(correct), len(wrong))

    if min_n < 15:
        print(f"  {label}: SKIP — only {min_n} balanced samples")
        return 0.5, 0.0

    rng = np.random.default_rng(42)
    correct = correct[rng.permutation(len(correct))[:min_n]]
    wrong = wrong[rng.permutation(len(wrong))[:min_n]]
    split = int(min_n * 0.8)
    tc, tw = correct[:split], wrong[:split]
    vc, vw = correct[split:], wrong[split:]

    # Small probe — question activations are 1 vector per question, so few samples
    if hidden_dim <= 1024:
        hdims = [128, 32]
    else:
        hdims = [256, 64]

    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=hidden_dim, hidden_dims=hdims, activation="silu")
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

    for _ in range(500):  # More epochs since fewer samples
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.003 * g, params, grads)
    set_p(ebm, params)

    n_eval = min(200, len(vc))
    ce = [float(ebm.energy(vc[i])) for i in range(n_eval)]
    we = [float(ebm.energy(vw[i])) for i in range(n_eval)]
    thresh = (np.mean(ce) + np.mean(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    acc = (tp + tn) / (len(ce) + len(we))
    gap = np.mean(we) - np.mean(ce)

    print(f"  {label}: test={acc:.1%}, gap={gap:.4f}, train={split}, test={min_n-split}")
    return acc, gap


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 27: Upstream Hallucination Detection")
    print("  Can we predict hallucination from the QUESTION alone?")
    print("=" * 70)

    # Test on smallest/fastest models first
    models = [
        ("LiquidAI/LFM2.5-350M", "lfm25-350m", False),
        ("Qwen/Qwen3.5-0.8B", "qwen35-08b", True),
        ("google/gemma-4-E2B-it", "gemma4-e2b-it", True),
    ]

    results = {}
    start = time.time()

    for model_name, short_id, has_chat in models:
        print(f"\n{'='*70}")
        print(f"MODEL: {short_id} ({model_name})")
        print(f"{'='*70}")

        acts, labels, hidden_dim, n_correct, n_wrong = collect_question_activations(
            model_name, short_id, has_chat, n_questions=400,
        )

        print(f"  Collected: {len(labels)} questions, {n_correct} correct, {n_wrong} wrong")
        print(f"  Question activation shape: ({len(acts)}, {hidden_dim})")

        # Train upstream detector
        print(f"\n  --- Question-Level Detection ---")
        q_acc, q_gap = train_and_evaluate(acts, labels, hidden_dim, "Question EBM")

        # Compare with per-token detection (from existing data)
        tok_file = os.path.join(DATA_DIR, f"token_activations_{short_id}_nothink.safetensors")
        if not os.path.exists(tok_file):
            tok_file = os.path.join(DATA_DIR, f"token_activations_{short_id}.safetensors")

        tok_acc = None
        if os.path.exists(tok_file):
            from safetensors.numpy import load_file
            tok_data = load_file(tok_file)
            print(f"\n  --- Per-Token Detection (baseline) ---")
            tok_acc, tok_gap = train_and_evaluate(
                tok_data["activations"], tok_data["labels"], hidden_dim, "Token EBM",
            )

        results[short_id] = {
            "model": model_name,
            "n_questions": len(labels),
            "model_accuracy": f"{n_correct}/{len(labels)} ({n_correct/len(labels)*100:.0f}%)",
            "question_ebm_acc": q_acc,
            "question_gap": q_gap,
            "token_ebm_acc": tok_acc,
        }

    # Summary
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 27 RESULTS: Upstream Detection ({elapsed:.0f}s)")
    print(sep)
    print(f"{'Model':20s} {'Model Acc':>10s} {'Question EBM':>13s} {'Token EBM':>10s} {'Delta':>8s}")
    print("-" * 65)

    for short_id, r in results.items():
        q = f"{r['question_ebm_acc']:.1%}"
        t = f"{r['token_ebm_acc']:.1%}" if r['token_ebm_acc'] is not None else "N/A"
        delta = ""
        if r['token_ebm_acc'] is not None:
            d = r['question_ebm_acc'] - r['token_ebm_acc']
            delta = f"{d:+.1%}"
        print(f"{short_id:20s} {r['model_accuracy']:>10s} {q:>13s} {t:>10s} {delta:>8s}")

    print(sep)

    # Verdict
    q_accs = [r["question_ebm_acc"] for r in results.values()]
    mean_q = np.mean(q_accs)
    if mean_q > 0.65:
        print(f"VERDICT: ✅ Upstream detection works! Mean question-level accuracy: {mean_q:.1%}")
        print("  Questions that will cause hallucination look different BEFORE generation.")
    elif mean_q > 0.55:
        print(f"VERDICT: ⚠️ Weak upstream signal. Mean: {mean_q:.1%}")
    else:
        print(f"VERDICT: ❌ No upstream signal. Mean: {mean_q:.1%}")
        print("  The model's representation of the question doesn't predict hallucination.")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
