#!/usr/bin/env python3
"""Experiment 25: Qwen3.5-0.8B with thinking DISABLED.

Hypothesis: Thinking mode (chain-of-thought before answering) may compress
hallucination signal further by making the model more "deliberate" and uniform.
Disabling thinking produces more direct responses that may have more
distinguishable correct/wrong activation patterns.

Compares:
  - With thinking (existing data): 67.2% EBM test accuracy
  - Without thinking: ?

Usage:
    sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        .venv/bin/python scripts/experiment_25_no_thinking.py'
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from collect_truthfulqa_activations import check_truthfulqa_answer


def main() -> int:
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import torch
    from datasets import load_dataset
    from safetensors.numpy import save_file

    from carnot.models.gibbs import GibbsConfig, GibbsModel
    from carnot.training.nce import nce_loss

    print("=" * 60)
    print("EXPERIMENT 25: Qwen3.5-0.8B WITHOUT Thinking")
    print("=" * 60)

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3.5-0.8B"
    print(f"\nLoading {model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        output_hidden_states=True,
        dtype=torch.float16 if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    # Load TruthfulQA (first 200 for speed, same as experiment 24)
    print("Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "generation")
    questions = list(ds["validation"])[:200]

    # === Phase 1: Collect activations WITHOUT thinking ===
    print("\n--- Phase 1: Collecting activations (NO THINKING) ---")

    all_token_ids = []
    all_activations = []
    all_labels = []
    n_correct_no_think = 0
    n_wrong_no_think = 0

    for qi, example in enumerate(questions):
        question = example["question"]
        correct_answers = example["correct_answers"]
        incorrect_answers = example["incorrect_answers"]
        best_answer = example.get("best_answer", "")

        # KEY DIFFERENCE: enable_thinking=False
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
        # Still strip any thinking tags just in case
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        is_correct = check_truthfulqa_answer(response, correct_answers, incorrect_answers, best_answer)
        if is_correct:
            n_correct_no_think += 1
        else:
            n_wrong_no_think += 1

        # Get hidden states
        with torch.no_grad():
            ho = model(outputs, output_hidden_states=True)
            hs = ho.hidden_states

        last_layer = hs[-1][0, prompt_len:, :].float().cpu().numpy()
        gen_tok_ids = gen_ids.cpu().numpy()

        for t in range(len(gen_tok_ids)):
            all_token_ids.append(int(gen_tok_ids[t]))
            all_activations.append(last_layer[t])
            all_labels.append(1 if is_correct else 0)

        if (qi + 1) % 50 == 0:
            print(f"  [{qi+1:3d}/{len(questions)}] "
                  f"tokens={len(all_token_ids)} "
                  f"correct={n_correct_no_think} wrong={n_wrong_no_think} "
                  f"({n_correct_no_think/(qi+1)*100:.0f}%)")

    n_tokens_no_think = len(all_token_ids)
    print(f"\nNo-thinking collection: {n_tokens_no_think} tokens")
    print(f"  Accuracy: {n_correct_no_think}/{len(questions)} ({n_correct_no_think/len(questions)*100:.0f}%)")

    # Save no-thinking activations for HuggingFace export
    nothink_file = os.path.join(os.path.dirname(__file__), "..", "data", "token_activations_qwen35_nothink.safetensors")
    from safetensors.numpy import save_file as sf_save
    sf_save(
        {
            "token_ids": np.array(all_token_ids, dtype=np.int32),
            "activations": np.stack(all_activations).astype(np.float32),
            "labels": np.array(all_labels, dtype=np.int32),
        },
        nothink_file,
    )
    print(f"  Saved to: {nothink_file} ({os.path.getsize(nothink_file) / 1e6:.1f} MB)")

    # === Phase 2: Train EBM on no-thinking activations ===
    print("\n--- Phase 2: Training per-token EBM ---")

    activations = jnp.array(np.array(all_activations))
    labels = np.array(all_labels)

    correct_mask = labels == 1
    wrong_mask = labels == 0
    correct_acts = activations[correct_mask]
    wrong_acts = activations[wrong_mask]

    min_n = min(len(correct_acts), len(wrong_acts))
    if min_n < 20:
        print(f"ERROR: Not enough balanced data (min={min_n})")
        return 1

    rng = np.random.default_rng(42)
    correct_acts = correct_acts[rng.permutation(len(correct_acts))[:min_n]]
    wrong_acts = wrong_acts[rng.permutation(len(wrong_acts))[:min_n]]
    split = int(min_n * 0.8)
    tc, tw = correct_acts[:split], wrong_acts[:split]
    vc, vw = correct_acts[split:], wrong_acts[split:]

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
        old = get_p(ebm)
        set_p(ebm, p)
        r = nce_loss(ebm, tc, tw)
        set_p(ebm, old)
        return r

    for ep in range(300):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.005 * g, params, grads)
    set_p(ebm, params)

    # Evaluate
    n_eval = min(300, len(vc))
    ce = [float(ebm.energy(vc[i])) for i in range(n_eval)]
    we = [float(ebm.energy(vw[i])) for i in range(n_eval)]
    thresh = (sum(ce) / len(ce) + sum(we) / len(we)) / 2
    tp = sum(1 for e in we if e > thresh)
    tn = sum(1 for e in ce if e <= thresh)
    test_acc_no_think = (tp + tn) / (len(ce) + len(we))
    gap = sum(we) / len(we) - sum(ce) / len(ce)

    # === Phase 3: Compare with thinking (from existing data) ===
    print("\n--- Phase 3: Compare with thinking-enabled data ---")

    # Load thinking-enabled activations from the merged dataset
    from safetensors.numpy import load_file
    merged_file = os.path.join(os.path.dirname(__file__), "..", "data", "token_activations_qwen35_merged.safetensors")

    if os.path.exists(merged_file):
        merged = load_file(merged_file)
        m_acts = jnp.array(merged["activations"])
        m_labels = merged["labels"]
        m_qids = merged["question_ids"]

        # Use TruthfulQA portion only (question_ids >= 10000) for apples-to-apples
        tqa_mask = m_qids >= 10000
        tqa_acts = m_acts[tqa_mask]
        tqa_labels = m_labels[tqa_mask]

        tqa_correct = tqa_acts[tqa_labels == 1]
        tqa_wrong = tqa_acts[tqa_labels == 0]
        tqa_min = min(len(tqa_correct), len(tqa_wrong))

        tqa_correct = tqa_correct[rng.permutation(len(tqa_correct))[:tqa_min]]
        tqa_wrong = tqa_wrong[rng.permutation(len(tqa_wrong))[:tqa_min]]
        tqa_split = int(tqa_min * 0.8)

        # Train separate EBM on thinking-enabled data
        tc2, tw2 = tqa_correct[:tqa_split], tqa_wrong[:tqa_split]
        vc2, vw2 = tqa_correct[tqa_split:], tqa_wrong[tqa_split:]

        key2 = jrandom.PRNGKey(42)
        ebm2 = GibbsModel(config, key=key2)
        params2 = get_p(ebm2)

        def loss_fn2(p):
            old = get_p(ebm2)
            set_p(ebm2, p)
            r = nce_loss(ebm2, tc2, tw2)
            set_p(ebm2, old)
            return r

        for ep in range(300):
            grads = jax.grad(loss_fn2)(params2)
            params2 = jax.tree.map(lambda p, g: p - 0.005 * g, params2, grads)
        set_p(ebm2, params2)

        n_eval2 = min(300, len(vc2))
        ce2 = [float(ebm2.energy(vc2[i])) for i in range(n_eval2)]
        we2 = [float(ebm2.energy(vw2[i])) for i in range(n_eval2)]
        thresh2 = (sum(ce2) / len(ce2) + sum(we2) / len(we2)) / 2
        tp2 = sum(1 for e in we2 if e > thresh2)
        tn2 = sum(1 for e in ce2 if e <= thresh2)
        test_acc_think = (tp2 + tn2) / (len(ce2) + len(we2))
        gap2 = sum(we2) / len(we2) - sum(ce2) / len(ce2)
    else:
        test_acc_think = 0.672  # From experiment 22
        gap2 = 0.0
        print("  (Using cached result: 67.2%)")

    # === Results ===
    sep = "=" * 60
    print(f"\n{sep}")
    print("EXPERIMENT 25 RESULTS: Thinking vs No-Thinking")
    print(sep)
    print(f"  Model: {model_name}")
    print(f"  Questions: {len(questions)} (TruthfulQA)")
    print(f"")
    print(f"  WITH thinking:")
    print(f"    EBM test accuracy: {test_acc_think:.1%}")
    print(f"    Energy gap:        {gap2:.4f}")
    print(f"")
    print(f"  WITHOUT thinking:")
    print(f"    Model accuracy:    {n_correct_no_think}/{len(questions)} ({n_correct_no_think/len(questions)*100:.0f}%)")
    print(f"    Tokens collected:  {n_tokens_no_think}")
    print(f"    EBM test accuracy: {test_acc_no_think:.1%}")
    print(f"    Energy gap:        {gap:.4f}")
    print(f"")

    delta = test_acc_no_think - test_acc_think
    if delta > 0.03:
        print(f"  VERDICT: ✅ No-thinking improves detection by {delta:.1%}")
        print(f"    Thinking mode DOES compress hallucination signal!")
    elif delta > 0:
        print(f"  VERDICT: ⚠️ Small improvement ({delta:+.1%})")
    elif delta > -0.03:
        print(f"  VERDICT: ⚠️ No significant difference ({delta:+.1%})")
    else:
        print(f"  VERDICT: ❌ No-thinking is worse ({delta:+.1%})")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
