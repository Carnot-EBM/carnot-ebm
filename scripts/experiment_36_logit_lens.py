#!/usr/bin/env python3
"""Experiment 36: Logit Lens Divergence as Hallucination Signal.

Project each layer's hidden states through the model's unembedding matrix
to see what word each layer "wants to output." If intermediate layers
predict different tokens than the final layer, that disagreement may
signal hallucination.

Key idea: the hallucination signal is in the DYNAMICS (how the model
changes its mind across layers), not the STATICS (final activation values).

Metrics per generated token:
  - Layer agreement: what fraction of layers predict the same top token as the final layer?
  - Max divergence: the layer with the most different prediction from final
  - Convergence point: at which layer does the model "commit" to its final answer?
  - Entropy trajectory: how does output entropy change across layers?

No EBM training needed — just forward passes and the model's own unembedding.

Usage:
    sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        PYTHONUNBUFFERED=1 .venv/bin/python scripts/experiment_36_logit_lens.py'
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


def logit_lens_analysis(
    model_name: str = "Qwen/Qwen3.5-0.8B",
    n_questions: int = 200,
) -> dict:
    """Run logit lens on generated tokens and compare correct vs wrong answers."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # Get the unembedding matrix (lm_head weights)
    # This projects hidden states → vocabulary logits
    if hasattr(model, 'lm_head'):
        unembed = model.lm_head.weight.data  # (vocab_size, hidden_dim)
    elif hasattr(model, 'embed_out'):
        unembed = model.embed_out.weight.data
    else:
        # Try to find it
        for name, param in model.named_parameters():
            if 'lm_head' in name or 'embed_out' in name:
                unembed = param.data
                break
        else:
            print("ERROR: Cannot find unembedding matrix")
            return {}

    print(f"  Unembedding shape: {unembed.shape}")  # (vocab_size, hidden_dim)
    n_layers = len([n for n, _ in model.named_modules() if 'layers.' in n and n.endswith('.mlp')])
    print(f"  Layers: ~{n_layers}")

    ds = load_dataset("truthful_qa", "generation")
    questions = list(ds["validation"])[:n_questions]

    # Collect per-token logit lens features
    correct_features = []
    wrong_features = []
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

        is_correct = check_truthfulqa_answer(
            response, correct_answers, incorrect_answers, best_answer,
        )
        if is_correct:
            n_correct += 1
        else:
            n_wrong += 1

        # Get hidden states from all layers
        with torch.no_grad():
            ho = model(outputs, output_hidden_states=True)
            hs = ho.hidden_states  # (n_layers+1,) tuple of (1, seq_len, hidden_dim)

        n_hs = len(hs)
        n_gen = len(gen_ids)

        # For each generated token, compute logit lens features
        for t_idx in range(n_gen):
            pos = prompt_len + t_idx
            actual_token = gen_ids[t_idx].item()

            # Project each layer's hidden state through unembedding
            layer_predictions = []
            layer_entropies = []
            layer_ranks_of_actual = []

            for layer_idx in range(n_hs):
                h = hs[layer_idx][0, pos, :]  # (hidden_dim,)

                # Project to vocabulary: logits = h @ unembed.T
                logits = h.float() @ unembed.float().T  # (vocab_size,)
                probs = torch.softmax(logits, dim=-1)

                # What does this layer predict?
                top_token = logits.argmax().item()
                layer_predictions.append(top_token)

                # Entropy of this layer's prediction distribution
                ent = -(probs * (probs + 1e-10).log()).sum().item()
                layer_entropies.append(ent)

                # Rank of the actual token in this layer's prediction
                sorted_indices = logits.argsort(descending=True)
                rank = (sorted_indices == actual_token).nonzero(as_tuple=True)[0]
                rank = rank[0].item() if len(rank) > 0 else len(logits)
                layer_ranks_of_actual.append(rank)

            # Compute features from the layer trajectories
            final_prediction = layer_predictions[-1]

            # 1. Layer agreement: what fraction predict the same as final?
            agreement = sum(1 for p in layer_predictions if p == final_prediction) / n_hs

            # 2. Convergence point: first layer where prediction matches final and stays
            convergence = n_hs  # default: never converges
            for l in range(n_hs):
                if all(p == final_prediction for p in layer_predictions[l:]):
                    convergence = l
                    break
            convergence_frac = convergence / n_hs

            # 3. Entropy trajectory: does entropy decrease monotonically?
            early_entropy = np.mean(layer_entropies[:n_hs // 3])
            mid_entropy = np.mean(layer_entropies[n_hs // 3: 2 * n_hs // 3])
            late_entropy = np.mean(layer_entropies[2 * n_hs // 3:])
            entropy_drop = early_entropy - late_entropy

            # 4. Rank stability: how stable is the actual token's rank across layers?
            rank_std = np.std(layer_ranks_of_actual)
            mean_rank = np.mean(layer_ranks_of_actual)
            final_rank = layer_ranks_of_actual[-1]

            # 5. Number of "mind changes": how many times does the top prediction change?
            mind_changes = sum(1 for i in range(1, n_hs) if layer_predictions[i] != layer_predictions[i - 1])

            features = {
                "agreement": agreement,
                "convergence_frac": convergence_frac,
                "early_entropy": early_entropy,
                "mid_entropy": mid_entropy,
                "late_entropy": late_entropy,
                "entropy_drop": entropy_drop,
                "rank_std": rank_std,
                "mean_rank": mean_rank,
                "final_rank": final_rank,
                "mind_changes": mind_changes,
            }

            if is_correct:
                correct_features.append(features)
            else:
                wrong_features.append(features)

        if (qi + 1) % 50 == 0:
            print(f"  [{qi+1:3d}/{n_questions}] correct={n_correct} wrong={n_wrong} "
                  f"tokens: {len(correct_features)}c/{len(wrong_features)}w")

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "correct": correct_features,
        "wrong": wrong_features,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
    }


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 36: Logit Lens Divergence as Hallucination Signal")
    print("  No EBM training — purely from the model's own computation")
    print("=" * 70)

    start = time.time()
    data = logit_lens_analysis("Qwen/Qwen3.5-0.8B", n_questions=200)

    if not data.get("correct") or not data.get("wrong"):
        print("ERROR: No data collected")
        return 1

    correct = data["correct"]
    wrong = data["wrong"]
    elapsed = time.time() - start

    print(f"\nCollected: {len(correct)} correct tokens, {len(wrong)} wrong tokens ({elapsed:.0f}s)")

    # Compare each feature between correct and wrong
    features = ["agreement", "convergence_frac", "entropy_drop", "mind_changes",
                 "early_entropy", "late_entropy", "rank_std", "mean_rank"]

    sep = "=" * 70
    print(f"\n{sep}")
    print("LOGIT LENS FEATURE COMPARISON: Correct vs Wrong")
    print(sep)
    print(f"{'Feature':25s} {'Correct':>10s} {'Wrong':>10s} {'Delta':>10s} {'Direction':>12s}")
    print("-" * 70)

    detection_results = {}

    for feat in features:
        c_vals = [f[feat] for f in correct]
        w_vals = [f[feat] for f in wrong]
        c_mean = np.mean(c_vals)
        w_mean = np.mean(w_vals)
        delta = w_mean - c_mean
        direction = "wrong higher" if delta > 0 else "correct higher"

        # Simple threshold detection accuracy
        all_vals = c_vals + w_vals
        all_labels = [0] * len(c_vals) + [1] * len(w_vals)
        thresh = (c_mean + w_mean) / 2

        if delta > 0:  # wrong is higher — classify above threshold as wrong
            tp = sum(1 for v in w_vals if v > thresh)
            tn = sum(1 for v in c_vals if v <= thresh)
        else:  # wrong is lower — classify below threshold as wrong
            tp = sum(1 for v in w_vals if v < thresh)
            tn = sum(1 for v in c_vals if v >= thresh)
        acc = (tp + tn) / (len(c_vals) + len(w_vals))

        detection_results[feat] = acc
        marker = " ***" if acc > 0.6 else ""
        print(f"  {feat:23s} {c_mean:>10.4f} {w_mean:>10.4f} {delta:>+10.4f} {direction:>12s}  {acc:.1%}{marker}")

    # Best feature
    best_feat = max(detection_results, key=detection_results.get)
    best_acc = detection_results[best_feat]

    print(f"\n  Best single feature: {best_feat} ({best_acc:.1%})")

    # Combined score: use all features with simple weighted sum
    print(f"\n--- Combined Score ---")
    # Normalize each feature and combine
    c_scores = []
    w_scores = []
    for i in range(len(correct)):
        score = 0
        for feat in features:
            c_mean = np.mean([f[feat] for f in correct])
            w_mean = np.mean([f[feat] for f in wrong])
            std = np.std([f[feat] for f in correct] + [f[feat] for f in wrong]) + 1e-10
            normalized = (correct[i][feat] - c_mean) / std
            # Weight by how well this feature discriminates
            weight = abs(w_mean - c_mean) / std
            score += weight * normalized
        c_scores.append(score)

    for i in range(len(wrong)):
        score = 0
        for feat in features:
            c_mean = np.mean([f[feat] for f in correct])
            w_mean = np.mean([f[feat] for f in wrong])
            std = np.std([f[feat] for f in correct] + [f[feat] for f in wrong]) + 1e-10
            normalized = (wrong[i][feat] - c_mean) / std
            weight = abs(w_mean - c_mean) / std
            score += weight * normalized
        w_scores.append(score)

    combined_thresh = (np.mean(c_scores) + np.mean(w_scores)) / 2
    if np.mean(w_scores) > np.mean(c_scores):
        tp = sum(1 for s in w_scores if s > combined_thresh)
        tn = sum(1 for s in c_scores if s <= combined_thresh)
    else:
        tp = sum(1 for s in w_scores if s < combined_thresh)
        tn = sum(1 for s in c_scores if s >= combined_thresh)
    combined_acc = (tp + tn) / (len(c_scores) + len(w_scores))
    print(f"  Combined logit lens score: {combined_acc:.1%}")

    # Compare to EBM baseline
    print(f"\n--- Comparison ---")
    print(f"  Logit lens best single:  {best_acc:.1%} ({best_feat})")
    print(f"  Logit lens combined:     {combined_acc:.1%}")
    print(f"  Per-token EBM (trained): 75.5% (requires model-specific training)")
    print(f"  Logit lens advantage:    NO training, NO domain specificity, NO model-specific EBM")

    if combined_acc > 0.755:
        print(f"\n  VERDICT: ✅ Logit lens BEATS trained EBM ({combined_acc:.1%} > 75.5%)!")
        print(f"  Zero training, cross-domain, from the model's own computation.")
    elif combined_acc > 0.65:
        print(f"\n  VERDICT: ⚠️ Logit lens shows promise ({combined_acc:.1%}) but below trained EBM")
    elif best_acc > 0.55:
        print(f"\n  VERDICT: ⚠️ Weak signal in {best_feat} ({best_acc:.1%})")
    else:
        print(f"\n  VERDICT: ❌ Logit lens features don't distinguish correct from wrong")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
